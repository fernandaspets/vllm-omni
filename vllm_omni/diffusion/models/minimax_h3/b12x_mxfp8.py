# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MXFP8 linears for the H3 DiT on b12x (SM120/SM121), opt-in via VLLM_OMNI_DIT_MXFP8=1.

The DiT's wide linears carry the step: at M=18,748 the FFN up-projection runs 29.22 ms in
bf16 and 8.54 ms through b12x's MXFP8 path with the activation quant fused (3.42x), weights
halved. The module keeps vLLM's parallel linear for weight loading and tensor parallel
sharding; only the math changes, on the local shard.

A plan is prepared per (capacity, shape) at first use. Capacity is bucketed and ``expected_m``
is deliberately NOT pinned, so a later request with a smaller token count reuses the same
program instead of recompiling.
"""
from __future__ import annotations

import os

import torch

_CAPACITY_STEP = 4096


def enabled() -> bool:
    return os.environ.get("VLLM_OMNI_DIT_MXFP8", "0") == "1"


def quantize_rows(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ModelOpt-style MXFP8 rows: 32-element blocks, E8M0 scale, e4m3 values."""
    rows, width = map(int, source.shape)
    blocked = source.to(torch.float32).reshape(rows, width // 32, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(max_abs > 0.0, max_abs / 448.0, torch.ones_like(max_abs))
    scale_u8 = (torch.ceil(torch.log2(safe)).clamp(-127, 127) + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    values = ((blocked / scale[..., None]).clamp(-448.0, 448.0)
              .to(torch.float8_e4m3fn).reshape(rows, width).contiguous())
    return values, scale_u8.contiguous()


def _capacity(n: int) -> int:
    return max(_CAPACITY_STEP, ((int(n) + _CAPACITY_STEP - 1) // _CAPACITY_STEP) * _CAPACITY_STEP)


class Mxfp8Linear:
    """Drop-in for a locally-sharded ``nn.Linear`` (weight [out, in]) on b12x MXFP8."""

    def __init__(self, weight: torch.Tensor, name: str = "h3-linear"):
        from b12x.gemm import mxfp8_linear

        self.name = name
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.packed = mxfp8_linear.pack_weight(*quantize_rows(weight.detach().to(torch.bfloat16)))
        self._plan = None
        self._capacity = 0

    def _prepare(self, capacity: int, device: torch.device) -> None:
        from b12x.gemm import blockscaled
        from b12x.preparation import PreparationSession, PreparedCall

        placeholder = torch.empty((capacity, self.in_features), dtype=torch.bfloat16,
                                  device=device)
        query = blockscaled.query_from_call(placeholder, self.packed, activation_mode="quantized")
        plan = blockscaled.plan(query)
        packed = self.packed

        def call(state):
            return PreparedCall(run=lambda: state.run(
                placeholder, packed.weight.values, packed.weight.scale_mma, None,
                activation_scale=None))

        with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
            session.prepare((plan.request(name=self.name, prepare_call=call),))
        self._plan = plan
        self._capacity = capacity

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from b12x.gemm import mxfp8_linear

        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if not flat.is_contiguous():
            flat = flat.contiguous()
        m = int(flat.shape[0])
        if self._plan is None or m > self._capacity:
            self._prepare(_capacity(m), flat.device)
        out = mxfp8_linear.mm(flat, self.packed, plan=self._plan)
        return out.reshape(*shape[:-1], self.out_features)
