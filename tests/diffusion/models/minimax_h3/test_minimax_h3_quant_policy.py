# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the H3 lane's per-role quantisation policy (``vllm_omni.diffusion.models.minimax_h3.quant_policy``).

The policy decides, per role, whether a wide DiT linear is swapped to NVFP4, kept at MXFP8,
or left as the original bf16 linear. Two failure modes were observed on the lane and are
pinned here:

1. The construction site passes the leaf in two spellings. ``role_for`` only stripped the
   dotted prefix after the fact; before that, ``attn.qkv_proj`` fell through to the
   ``refiner`` default, so a boot quantised *nothing* while still loading and serving
   normally (lane log: ``mxfp8=0 nvfp4=0 bf16_roles=400``).
2. The control file is hand-edited between boots, so a typo must never kill a load.

The module is stdlib-only by design, so no GPU and no torch are needed.
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.minimax_h3 import quant_policy as h3_quant_policy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _force_policy_reread():
    """The module caches the policy for 1 s; force every read to re-resolve."""
    h3_quant_policy._cache["t"] = 0.0
    yield
    h3_quant_policy._cache["t"] = 0.0


def _write_policy(monkeypatch, tmp_path, text):
    control = tmp_path / "QUANT_POLICY"
    control.write_text(text)
    monkeypatch.setenv("H3_QUANT_POLICY_CONTROL", str(control))
    return h3_quant_policy.current()


# --- leaf -> role mapping ------------------------------------------------------------


def test_role_for_accepts_the_bare_role():
    assert h3_quant_policy.role_for("qkv_proj") == "attn"
    assert h3_quant_policy.role_for("out_proj") == "attn"
    assert h3_quant_policy.role_for("fc1") == "mlp"
    assert h3_quant_policy.role_for("fc2") == "mlp"


@pytest.mark.parametrize(
    "leaf",
    [
        "attn.qkv_proj",
        "attn.out_proj",
        "mlp.fc1",
        "mlp.fc2",
        "blocks.0.attn.qkv_proj",
        "blocks.47.mlp.fc2",
    ],
)
def test_role_for_strips_the_dotted_prefix(leaf):
    """Regression: the dotted spelling must not fall through to the refiner default."""
    assert h3_quant_policy.role_for(leaf) == h3_quant_policy.role_for(leaf.rsplit(".", 1)[-1])
    assert h3_quant_policy.role_for(leaf) in {"attn", "mlp"}


@pytest.mark.parametrize("leaf", ["to_gate_compress", "refiner.weight", "", "unknown"])
def test_role_for_unknown_leaf_is_refiner(leaf):
    assert h3_quant_policy.role_for(leaf) == "refiner"


# --- the built-in default ------------------------------------------------------------


def test_default_policy_is_the_documented_middle_ground():
    assert h3_quant_policy.DEFAULT_POLICY == {
        "mlp": "nvfp4",
        "attn": "mxfp8",
        "refiner": "bf16",
    }


def test_missing_control_file_falls_back_to_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_QUANT_POLICY_CONTROL", str(tmp_path / "does_not_exist"))
    assert h3_quant_policy.current() == h3_quant_policy.DEFAULT_POLICY


# --- control-file parsing ------------------------------------------------------------


def test_control_file_overrides_the_default(monkeypatch, tmp_path):
    policy = _write_policy(monkeypatch, tmp_path, "mlp mxfp8\nattn bf16\nrefiner nvfp4\n")
    assert policy == {"mlp": "mxfp8", "attn": "bf16", "refiner": "nvfp4"}


def test_control_file_ignores_comments_blanks_and_malformed_lines(monkeypatch, tmp_path):
    policy = _write_policy(
        monkeypatch,
        tmp_path,
        "# a comment\n\nmlp nvfp4\n  \nnot a pair at all\nattn mxfp8 extra tokens\nrefiner bf16\n",
    )
    assert policy == h3_quant_policy.DEFAULT_POLICY


def test_control_file_ignores_unknown_dtype(monkeypatch, tmp_path):
    policy = _write_policy(monkeypatch, tmp_path, "mlp fp8\nrefiner bf16\n")
    assert policy["mlp"] == h3_quant_policy.DEFAULT_POLICY["mlp"]
    assert policy["refiner"] == "bf16"


def test_control_file_ignores_unknown_role(monkeypatch, tmp_path):
    """A typo'd role must be dropped, not silently added to the active policy."""
    policy = _write_policy(monkeypatch, tmp_path, "attn nvfp4\nattnn nvfp4\nmlp bf16\n")
    assert "attnn" not in policy
    assert policy["attn"] == "nvfp4"
    assert policy["mlp"] == "bf16"


def test_control_file_is_re_read_after_the_cache_expires(monkeypatch, tmp_path):
    control = tmp_path / "QUANT_POLICY"
    control.write_text("mlp mxfp8\n")
    monkeypatch.setenv("H3_QUANT_POLICY_CONTROL", str(control))
    assert h3_quant_policy.current()["mlp"] == "mxfp8"

    control.write_text("mlp nvfp4\n")
    h3_quant_policy._cache["t"] = 0.0  # simulate the 1 s TTL elapsing
    assert h3_quant_policy.current()["mlp"] == "nvfp4"


# --- the accessors the model actually calls ------------------------------------------


def test_policy_for_follows_the_role_mapping(monkeypatch, tmp_path):
    _write_policy(monkeypatch, tmp_path, "attn bf16\nmlp nvfp4\nrefiner mxfp8\n")
    assert h3_quant_policy.policy_for("qkv_proj") == "bf16"
    assert h3_quant_policy.policy_for("attn.qkv_proj") == "bf16"
    assert h3_quant_policy.policy_for("mlp.fc1") == "nvfp4"
    assert h3_quant_policy.policy_for("to_gate_compress") == "mxfp8"


def test_policy_for_never_returns_an_unsupported_dtype(monkeypatch, tmp_path):
    _write_policy(monkeypatch, tmp_path, "mlp nvfp4\nattn mxfp8\nrefiner bf16\n")
    for leaf in ("qkv_proj", "out_proj", "fc1", "fc2", "attn.qkv_proj", "blocks.1.mlp.fc1"):
        assert h3_quant_policy.policy_for(leaf) in {"nvfp4", "mxfp8", "bf16"}


def test_describe_names_every_role(monkeypatch, tmp_path):
    _write_policy(monkeypatch, tmp_path, "mlp nvfp4\nattn mxfp8\nrefiner bf16\n")
    described = h3_quant_policy.describe()
    for role in h3_quant_policy.DEFAULT_POLICY:
        assert role in described
