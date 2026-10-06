"""Batch the Ulysses q/k/v/gate all-to-all into ONE collective (control-file gated).

Why: the H3 DiT issues four separate all_to_all calls per block for q, k, v and the VSA
gate_compress (4 x 50 blocks = the 200 `nccl:all_to_all` calls/step in the step profile). The
four tensors are identically shaped (56 heads x 128, no GQA in this DiT), and `all_to_all_4D`
only transposes the head (scatter) and sequence (gather) dims while dim 0 (batch) rides along
untouched, so all four can be exchanged as one stacked batch.

Equivalence is by construction (the element mapping per batch row is unchanged), and it was
verified bit-identical against the stock quartet at the real geometry by
test_a2a_qkv_batch.py: `exact=True maxabs=0.000e+00` in both directions.

Arm selection: H3_A2A_QKV_BATCH_CONTROL names a file holding 0/1 (default from
H3_A2A_QKV_BATCH). The file is re-read at most once per second, so a boot can A/B both arms
without a restart, and a missing/unreadable file can only fall back to stock.

Numerics note: under the int8 wire arm the batched call computes one activation scale for the
stacked batch instead of one per tensor, so the int8 result is NOT bit-identical; the bf16 arm is.

Imported through the env-gated branch inserted into
vllm_omni/diffusion/attention/parallel/ulysses.py by patch_h3_a2a_qkv_batch.py.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence

import torch

from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

logger = logging.getLogger(__name__)

__all__ = ["enabled", "batched_seq_a2a"]

_CONTROL = os.environ.get("H3_A2A_QKV_BATCH_CONTROL", "")
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
    if _CONTROL:
        try:
            with open(_CONTROL) as fh:
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
