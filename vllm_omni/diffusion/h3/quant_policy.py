"""Dynamic per-role quantisation policy for the MiniMax-H3 DiT wide linears.

Why this exists
---------------
The two existing arms are all-or-nothing: `VLLM_OMNI_DIT_MXFP8=1` swaps all 200 wide
linears (50 blocks x {attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}) onto b12x e4m3,
and on top of that `VLLM_OMNI_DIT_NVFP4=1` swaps the same 200 onto b12x FP4 (W4A4).
The full NVFP4 swap is the fastest arm measured here (-1.92 s/clip on 4 steps) but it
changes the precision of the attention projections as well as the MLP, i.e. it stacks
the whole quality surface at once.

This module lets one boot choose **per role**:

    nvfp4  -> b12x FP4 weight + per-call activation global scale (fastest, lowest precision)
    mxfp8  -> b12x e4m3 weight                                    (middle)
    bf16   -> leave the original vLLM linear untouched            (highest precision, no swap)

The default policy is the documented middle ground: the MLP (`fc1`+`fc2`) takes NVFP4
because it is ~63% of the measured NVFP4 win, while the attention projections
(`qkv_proj`+`out_proj`) stay MXFP8. Refiner/AdaLN paths stay bf16.

No-swap is free on the model side: the DiT forwards read
`self._mx_qkv(x) if self._mx_qkv is not None else self.qkv_proj(x)`, so "bf16" simply
means the attribute is never set.

Control file (so a single boot can A/B without rebooting)
--------------------------------------------------------
`H3_QUANT_POLICY_CONTROL` names a file of lines `<role> <dtype>`:

    mlp nvfp4
    attn mxfp8
    refiner bf16

The file is re-read at most once per second and an unreadable/missing file means the
built-in default. Unknown roles and unknown dtypes are ignored (never crash a load).
"""

from __future__ import annotations

import os
import time

# role -> dtype, the built-in default (the measured middle ground)
DEFAULT_POLICY: dict[str, str] = {
    "mlp": "nvfp4",     # fc1 + fc2: ~63% of the measured NVFP4 win
    "attn": "mxfp8",    # qkv_proj + out_proj: keep the sensitive projections high precision
    "refiner": "bf16",  # token_refiner blocks + anything else: untouched
}
_VALID = {"nvfp4", "mxfp8", "bf16"}
_LEAF_ROLE = {
    "qkv_proj": "attn",
    "out_proj": "attn",
    "fc1": "mlp",
    "fc2": "mlp",
}

_TTL = 1.0
_cache: dict[str, object] = {"t": 0.0, "policy": dict(DEFAULT_POLICY), "src": "default"}


def _control_path() -> str:
    """Path to the optional control file; unset means "use DEFAULT_POLICY"."""
    return os.environ.get("H3_QUANT_POLICY_CONTROL", "")


def current() -> dict[str, str]:
    """Return the active policy, re-reading the control file at most once per second."""
    now = time.monotonic()
    if now - float(_cache["t"]) < _TTL:  # type: ignore[arg-type]
        return _cache["policy"]  # type: ignore[return-value]
    policy = dict(DEFAULT_POLICY)
    src = "default"
    path = _control_path()
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                role, dtype = parts[0].strip().lower(), parts[1].strip().lower()
                if dtype not in _VALID or role not in DEFAULT_POLICY:
                    continue  # unknown role or dtype: ignore, never crash a load
                policy[role] = dtype
        src = path
    except OSError:
        pass  # missing/unreadable control file -> default policy
    _cache["t"], _cache["policy"], _cache["src"] = now, policy, src
    return policy


def role_for(leaf: str) -> str:
    """Map a construction-site leaf name to a policy role.

    Accepts either a bare role ('qkv_proj') or a dotted leaf ('attn.qkv_proj'), because the
    construction site passes both spellings; getting this wrong silently made every role fall
    through to 'refiner' and disabled *all* quantisation (measured 2026-10-04: mxfp8=0 nvfp4=0).
    """
    key = leaf.rsplit(".", 1)[-1]
    return _LEAF_ROLE.get(key, "refiner")


def policy_for(leaf: str) -> str:
    """Quantisation dtype wanted for this leaf: 'nvfp4' | 'mxfp8' | 'bf16'."""
    return current().get(role_for(leaf), "mxfp8")


def describe() -> str:
    """One-line summary for the lane log (proves which policy a boot actually ran)."""
    pol = current()
    return f"policy={_cache['src']} " + " ".join(f"{r}={pol[r]}" for r in sorted(pol))
