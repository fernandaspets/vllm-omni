# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_SAGE_ATTN_TENSOR_LAYOUT = os.environ.get("SAGE_ATTN_TENSOR_LAYOUT", "NHD").upper()
assert _SAGE_ATTN_TENSOR_LAYOUT in ("NHD", "HND"), (
    f"SAGE_ATTN_TENSOR_LAYOUT must be 'NHD' or 'HND', got '{_SAGE_ATTN_TENSOR_LAYOUT}'"
)

if current_omni_platform.is_xpu():
    try:
        import inspect

        from auto_round_kernel import ARK

        _ark = ARK()
        xpu_sageattn = _ark.sagev1
        _sagev1_params = inspect.signature(xpu_sageattn).parameters
        _sagev1_has_tensor_layout = "tensor_layout" in _sagev1_params
        _sagev1_scale_param = "sm_scale" if "sm_scale" in _sagev1_params else "scale"
    except ImportError:
        logger.warning(
            "XPU SageAttention (auto_round_kernel.ARK.sagev1) is not available. "
            "Install auto-round-lib for XPU sage attention support."
        )
        xpu_sageattn = None
        _sagev1_has_tensor_layout = False
        _sagev1_scale_param = "scale"
else:
    try:
        from sageattention import sageattn, sageattn_varlen
    except ImportError:
        logger.warning(
            "SageAttentionBackend is not available. You may install sage-attention"
            " by pip install git+https://github.com/thu-ml/SageAttention.git"
        )
        sageattn = None
        sageattn_varlen = None

# TODO add sage3 attention backend


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # forward_cuda dispatches sageattn_varlen
        # over the caller's packed cu_seqlens, so a packed sequence with padding
        # does not need a boolean attn_mask (the mask is what SAGE cannot take).
        return True

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        if backend_kwargs:
            logger.warning("SageAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        packed = getattr(attn_metadata, "packed_padding", None) if attn_metadata is not None else None
        if packed is not None:
            if sageattn_varlen is None:
                raise ImportError(
                    "SAGE_ATTN requires sageattention. Install with: "
                    "pip install git+https://github.com/thu-ml/SageAttention.git"
                )
            # The packed buffer carries uninitialised padding beyond q_length
            # (the cuDNN path slices K/V itself; flash only reads the declared
            # ranges). Sage's Triton kernel validates the whole tensor, so slice
            # to the declared lengths, run, then scatter back into a zero-filled
            # buffer of the original shape.
            q3 = query.flatten(0, 1)
            k3 = key.flatten(0, 1)
            v3 = value.flatten(0, 1)
            n = int(packed.q_length)
            out = sageattn_varlen(
                q3[:n],
                k3[:n],
                v3[:n],
                packed.cu_seqlens_q,
                packed.cu_seqlens_k,
                n,
                int(packed.kv_length),
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )
            if out.shape[0] == q3.shape[0]:
                return out.reshape_as(query)
            full = q3.new_zeros((q3.shape[0],) + tuple(out.shape[1:]))
            full[: out.shape[0]] = out
            return full.reshape_as(query)
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            raise ValueError("SAGE_ATTN does not support attn_mask. Select a mask-capable backend.")
        if sageattn is None:
            raise ImportError(
                "SAGE_ATTN requires sageattention. Install with: "
                "pip install git+https://github.com/thu-ml/SageAttention.git"
            )
        output = sageattn(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
        )
        return output

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if xpu_sageattn is None:
            raise ImportError("XPU SageAttention requires auto-round-lib. Install with: pip install auto-round-lib")
        orig_dtype = query.dtype
        q = query.to(torch.float16) if orig_dtype != torch.float16 else query
        k = key.to(torch.float16) if orig_dtype != torch.float16 else key
        v = value.to(torch.float16) if orig_dtype != torch.float16 else value

        if _sagev1_has_tensor_layout:
            if _SAGE_ATTN_TENSOR_LAYOUT == "HND":
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
            output = xpu_sageattn(
                q,
                k,
                v,
                tensor_layout=_SAGE_ATTN_TENSOR_LAYOUT,
                is_causal=self.causal,
                **{_sagev1_scale_param: self.softmax_scale},
            )
            if _SAGE_ATTN_TENSOR_LAYOUT == "HND":
                output = output.transpose(1, 2).contiguous()
        else:
            # No tensor_layout support: kernel expects HND [B, H, S, D]
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            output = xpu_sageattn(
                q,
                k,
                v,
                is_causal=self.causal,
                **{_sagev1_scale_param: self.softmax_scale},
            )
            output = output.transpose(1, 2).contiguous()

        if orig_dtype != torch.float16:
            output = output.to(orig_dtype)
        return output
