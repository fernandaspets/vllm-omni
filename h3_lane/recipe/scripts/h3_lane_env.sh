#!/bin/bash
# h3_lane_env.sh - the lane's arm selector. SOURCE this, do not execute it.
#
#   H3_QUANT=mxfp8|hybrid|nvfp4     (default mxfp8)
#   H3_STEPS=2|4|8                  (default 4)
#   H3_WEIGHTS_SOURCE=local|hf      (default local)
#
# One place decides every arm, validates it, logs what it chose, and fails loudly on an
# unknown value. Nothing here is implicit: a typo is an error, not a silent fallback.
#
#   H3_PRINT=1   print the resolved configuration and exit without touching anything
#   H3_HF_DRY=1  with H3_WEIGHTS_SOURCE=hf, print what would be fetched instead of fetching
#
# Provenance: the quant policy mechanics live in port/h3_quant_policy.py (control file
# H3_QUANT_POLICY_CONTROL, lines "<role> <dtype>", roles mlp/attn/refiner). The MXFP8 path is
# gated by VLLM_OMNI_DIT_MXFP8, the NVFP4 class by VLLM_OMNI_DIT_NVFP4; a policy asking for nvfp4
# while NVFP4 is off is downgraded with a warning, which is why this script sets both together
# rather than relying on that fallback.

set -euo pipefail

H3_QUANT=${H3_QUANT:-mxfp8}
H3_STEPS=${H3_STEPS:-4}
H3_WEIGHTS_SOURCE=${H3_WEIGHTS_SOURCE:-local}
H3_ARM_DIR=${H3_ARM_DIR:-/mnt/2king/build/h3/research/2026-10-03-step-profile/arms}
H3_REFMOD_DIR=${H3_REFMOD_DIR:-/home/giga/refmods_ro}

_h3_die() { echo "[h3_lane_env] ERROR: $*" >&2; exit 2; }

# ---------------------------------------------------------------- quantisation arm
case "$H3_QUANT" in
  mxfp8)
    # the standing default: every linear in MXFP8, refiner left in bf16
    export VLLM_OMNI_DIT_MXFP8=1
    export VLLM_OMNI_DIT_NVFP4=0
    _h3_policy="mlp mxfp8
attn mxfp8
refiner bf16"
    ;;
  hybrid)
    # per-role: FFN in NVFP4, attention projections kept in MXFP8
    export VLLM_OMNI_DIT_MXFP8=1
    export VLLM_OMNI_DIT_NVFP4=1
    _h3_policy="mlp nvfp4
attn mxfp8
refiner bf16"
    ;;
  nvfp4)
    # full W4A4: every role that supports it goes NVFP4
    export VLLM_OMNI_DIT_MXFP8=0
    export VLLM_OMNI_DIT_NVFP4=1
    _h3_policy="mlp nvfp4
attn nvfp4
refiner bf16"
    ;;
  *) _h3_die "unknown H3_QUANT='$H3_QUANT' (want mxfp8|hybrid|nvfp4)" ;;
esac

# Write the policy to a per-arm file so the shared control files are never mutated, and point the
# resolver at it. The policy is what the model actually resolves; the env above only enables the
# classes, so both must agree for the arm to mean what its name says.
mkdir -p "$H3_ARM_DIR/$H3_QUANT"
_h3_policy_file="$H3_ARM_DIR/$H3_QUANT/QUANT_POLICY"
printf '%s\n' "$_h3_policy" > "$_h3_policy_file"
export H3_QUANT_POLICY_CONTROL="$_h3_policy_file"

# ---------------------------------------------------------------- step arm
case "$H3_STEPS" in
  2)
    H3_LORA=${H3_LORA:-/mnt/2king/models/loras_pdmd/minimax_h3_ref2v_turbo_2step_v1.0_bf16.safetensors}
    _h3_steps_note="2-step distilled (PDMD) adapter; rejected on quality by the maintainer, kept as an arm"
    ;;
  4)
    H3_LORA=${H3_LORA:-/mnt/2king/models/lightx2v/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors}
    _h3_steps_note="4-step turbo v0.1 - the standing default"
    ;;
  8)
    H3_LORA=${H3_LORA:-/mnt/2king/models/lightx2v/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors}
    _h3_steps_note="8-step turbo v1.0 (768p variant); confirm the resolution/shift pairing before trusting a comparison"
    ;;
  *) _h3_die "unknown H3_STEPS='$H3_STEPS' (want 2|4|8)" ;;
esac
export H3_LORA
# The step count itself is a per-request parameter; export it so request drivers use the same arm.
export H3_REQUEST_STEPS="$H3_STEPS"

# ---------------------------------------------------------------- weights source
case "$H3_WEIGHTS_SOURCE" in
  local)
    MODEL=${MODEL:-/mnt/2king/models/MiniMaxAI/MiniMax-H3-ref2va}
    export H3_REFMOD_PATHS=${H3_REFMOD_PATHS-$H3_REFMOD_DIR/minimaxh3_rocco_v2_refmod.safetensors:$H3_REFMOD_DIR/minimaxh3_roxy_v2_refmod.safetensors}
    ;;
  hf)
    # Resolved into a local cache; the layouts differ from the local tree, so the helper builds
    # the ref2va wrapper (model_index.json + Ref2VA symlink) and returns the directory to serve.
    if [ "${H3_HF_DRY:-0}" = "1" ]; then
      MODEL="<hf:${H3_HF_MODEL_REPO:-MiniMaxAI/MiniMax-H3} -> ref2va wrapper>"
    else
      MODEL=$(python3 "$(dirname "${BASH_SOURCE[0]}")/h3_fetch_weights.py" --steps "$H3_STEPS")
    fi
    # The Rocco/Roxy RefMods are private working assets: they are never published and never
    # fetched. An HF user therefore gets no identity RefMods, and the run says so out loud.
    export H3_REFMOD_PATHS=""
    echo "[h3_lane_env] NOTE: H3_REFMOD_PATHS is empty - identity RefMods are private and are not"
    echo "[h3_lane_env]       downloaded from anywhere. Renders will not carry the Rocco/Roxy identity."
    ;;
  *) _h3_die "unknown H3_WEIGHTS_SOURCE='$H3_WEIGHTS_SOURCE' (want local|hf)" ;;
esac
export MODEL
export H3_REFMOD_NORMALIZE=${H3_REFMOD_NORMALIZE:-1}

# ---------------------------------------------------------------- report
echo "[h3_lane_env] arm: quant=$H3_QUANT steps=$H3_STEPS weights=$H3_WEIGHTS_SOURCE"
echo "[h3_lane_env]   policy file : $H3_QUANT_POLICY_CONTROL"
echo "[h3_lane_env]   policy      : $(tr '\n' ' ' < "$_h3_policy_file")"
echo "[h3_lane_env]   dit gates   : MXFP8=$VLLM_OMNI_DIT_MXFP8 NVFP4=$VLLM_OMNI_DIT_NVFP4"
echo "[h3_lane_env]   lora        : $H3_LORA"
echo "[h3_lane_env]   lora note   : $_h3_steps_note"
echo "[h3_lane_env]   model       : $MODEL"
echo "[h3_lane_env]   refmods     : ${H3_REFMOD_PATHS:-<none>}"
echo "[h3_lane_env]   request steps: $H3_REQUEST_STEPS"

if [ "${H3_PRINT:-0}" = "1" ]; then
  echo "[h3_lane_env] H3_PRINT=1 -> the caller should stop here (nothing started)"
  H3_PRINT_DONE=1
fi
