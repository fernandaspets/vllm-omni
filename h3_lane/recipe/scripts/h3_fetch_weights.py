#!/usr/bin/env python3
"""Resolve MiniMax-H3 weights from Hugging Face for the ref2va task.

Mirrors the known-good local layout rather than inventing one:

    <cache>/MiniMax-H3/           snapshot of MiniMaxAI/MiniMax-H3
    <cache>/ref2va/
        model_index.json          the ref2va partition config (shipped in this bundle)
        Ref2VA -> ../MiniMax-H3/Ref2VA

Prints the directory to serve on stdout, so the caller can do MODEL=$(...).

What it deliberately does NOT do: fetch identity RefMods. Those are private working assets
and are never published; the caller clears H3_REFMOD_PATHS and says so.

The full base repo is large (order 250+ GB), so this is not exercised in CI; --dry-run prints
the plan and the exact repositories/patterns without downloading anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_REPO = os.environ.get("H3_HF_MODEL_REPO", "MiniMaxAI/MiniMax-H3")
LORA_REPO = os.environ.get("H3_HF_LORA_REPO", "lightx2v/Minimax-h3-Turbo")

# The turbo adapters Live in one HF repo under different filenames; map the arm to the file.
LORA_BY_STEPS = {
    "2": os.environ.get("H3_HF_LORA_2STEP", "minimax_h3_ref2v_turbo_2step_v1.0_bf16.safetensors"),
    "4": os.environ.get("H3_HF_LORA_4STEP", "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"),
    "8": os.environ.get("H3_HF_LORA_8STEP", "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"),
}

HERE = Path(__file__).resolve().parent
WRAPPER_CONFIG = HERE / "ref2va" / "model_index.json"


def plan(steps: str, cache: Path) -> dict:
    return {
        "model_repo": MODEL_REPO,
        "lora_repo": LORA_REPO,
        "lora_file": LORA_BY_STEPS[steps],
        "cache": str(cache),
        "snapshot": str(cache / "MiniMax-H3"),
        "wrapper": str(cache / "ref2va"),
        "serve_dir": str(cache / "ref2va"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=os.environ.get("H3_STEPS", "4"), choices=["2", "4", "8"])
    ap.add_argument("--cache", default=os.environ.get(
        "H3_HF_CACHE", str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "h3-ref2va")))
    ap.add_argument("--dry-run", action="store_true", default=os.environ.get("H3_HF_DRY") == "1")
    ap.add_argument("--print-lora", action="store_true", help="print the LoRA path for the arm")
    ap.add_argument("--verify-only", action="store_true", help="check an existing cache, do not download")
    args = ap.parse_args()

    cache = Path(args.cache)
    p = plan(args.steps, cache)
    lora_path = cache / "loras" / p["lora_file"]

    if args.dry_run:
        print(json.dumps({"dry_run": True, **p, "lora_path": str(lora_path),
                          "refmods": "not fetched (private)"}, indent=2), file=sys.stderr)
        print(p["serve_dir"])
        return 0

    if args.verify_only:
        missing = [x for x in (cache / "MiniMax-H3", p["wrapper"], lora_path) if not x.exists()]
        if missing:
            print("missing: " + ", ".join(map(str, missing)), file=sys.stderr)
            return 1
        print(lora_path if args.print_lora else p["serve_dir"])
        return 0

    if not WRAPPER_CONFIG.is_file():
        print(f"missing shipped wrapper config: {WRAPPER_CONFIG}", file=sys.stderr)
        return 1

    from huggingface_hub import hf_hub_download, snapshot_download

    # 1. base model
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(cache / "MiniMax-H3"))

    # 2. wrapper, laid out exactly like the local tree
    wrapper = cache / "ref2va"
    wrapper.mkdir(parents=True, exist_ok=True)
    (wrapper / "model_index.json").write_text(WRAPPER_CONFIG.read_text())
    ref2va = wrapper / "Ref2VA"
    target = Path("..") / "MiniMax-H3" / "Ref2VA"
    if ref2va.is_symlink() or ref2va.exists():
        ref2va.unlink()
    ref2va.symlink_to(target)

    # 3. the turbo adapter for this arm
    hf_hub_download(repo_id=LORA_REPO, filename=p["lora_file"], local_dir=str(cache / "loras"))

    print(lora_path if args.print_lora else p["serve_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
