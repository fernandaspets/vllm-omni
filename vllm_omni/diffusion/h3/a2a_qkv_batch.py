"""Batch the Ulysses q/k/v/gate all-to-all into ONE collective (control-file gated).

Why: the H3 DiT issues four separate all_to_all calls per block for q, k, v and the VSA
gate_compress (4 x 50 blocks = the 200 `nccl:all_to_all` calls/step in the step profile). The
four tensors are identically shaped (56 heads x 128, no GQA in this DiT), and `all_to_all_4D`
only transposes the head (scatter) and sequence (gather) dims while dim 0 (batch) rides along
untouched, so all four can be exchanged as one stacked batch.

Equivalence is by construction: the element mapping per batch row is unchanged, so a bf16 batch is
the concatenation of the four stock exchanges.

Arm selection: H3_A2A_QKV_BATCH_CONTROL names a file holding 0/1 (default from
H3_A2A_QKV_BATCH). The file is re-read at most once per second, so a boot can A/B both arms
without a restart, and a missing/unreadable file can only fall back to stock.

Numerics note: under the int8 wire arm the batched call computes one activation scale for the
stacked batch instead of one per tensor, so the int8 result is NOT bit-identical; the bf16 arm is.

Registered with vllm_omni/diffusion/attention/parallel/ulysses.py by the H3 model at setup.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence

import torch

from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

logger = logging.getLogger(__name__)

__all__ = ["enabled", "batched_seq_a2a", "install"]

_CACHE_TTL = 1.0
_cached: tuple[float, bool] | None = None
_LOGGED: set = set()


def enabled() -> bool:
    """True when batching is selected (control file first, else H3_A2A_QKV_BATCH)."""
    global _cached
    now = time.monotonic()
    if _cached is not None and now - _cached[0] < _CACHE_TTL:
        return _cached[1]
    flag = os.environ.get("H3_A2A_QKV_BATCH", "0") == "1"
    control = os.environ.get("H3_A2A_QKV_BATCH_CONTROL", "")
    if control:
        try:
            with open(control) as fh:
                flag = fh.read().strip() == "1"
        except OSError:
            flag = False  # unreadable control file can only ever mean stock
    _cached = (now, flag)
    return flag


def _note_engaged(n: int, shape: Sequence[int]) -> None:
    """One deduplicated line per stacked shape (ints only - never format a tensor value)."""
    key = (n, tuple(int(x) for x in shape))
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.info("h3_a2a_qkv_batch engaged: stacked %d tensors shape=%s", n, key[1])


def batched_seq_a2a(
    group,
    tensors: Sequence[torch.Tensor],
    scatter_idx: int,
    gather_idx: int,
    use_sync: bool,
) -> list[torch.Tensor]:
    """Reshard N identically-shaped (B, S, H, D) tensors with a single all-to-all.

    Stacks on dim 0, runs the stock SeqAllToAll4D once, slices the result back along dim 0
    (the head/seq dims keep their indices, so scatter_idx/gather_idx are unchanged). Falls
    back to one call per tensor if the shapes differ or the group is trivial.
    """
    if not tensors:
        return []
    first = tensors[0]
    same_shape = all(t.shape == first.shape for t in tensors[1:])
    if len(tensors) == 1 or not same_shape:
        return [SeqAllToAll4D.apply(group, t, scatter_idx, gather_idx, use_sync) for t in tensors]

    bsz = int(first.shape[0])
    stacked = torch.cat(list(tensors), dim=0)
    _note_engaged(len(tensors), stacked.shape)
    out = SeqAllToAll4D.apply(group, stacked, scatter_idx, gather_idx, use_sync)
    # Row-major slices of a contiguous tensor are contiguous, so this is a view, not a copy.
    return [out[i * bsz : (i + 1) * bsz] for i in range(len(tensors))]


def _registered_adapter(group, tensors, scatter_idx, gather_idx, use_sync):
    """Adapter the generic Ulysses path calls; None means "not selected, use stock"."""
    if not enabled():
        return None
    return batched_seq_a2a(group, tensors, scatter_idx, gather_idx, use_sync)


def install() -> None:
    """Register this batching with the generic parallel-attention layer."""
    from vllm_omni.diffusion.attention.parallel.ulysses import register_batched_qkv_exchange

    register_batched_qkv_exchange(_registered_adapter)
