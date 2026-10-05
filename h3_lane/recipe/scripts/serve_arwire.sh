#!/bin/bash
# ref2av lane INSIDE the karmic image (torch 2.14 / CUDA 13.4):
#   body=SOL_ATTN, refiner=B12X (ours; dense mode -- SAGE is not in this image).
# AR-wire arm: durable config + BOTH wire transports (a2a + TP all-reduce), driven at runtime by
# control files so one boot can A/B all arms. Env defaults are bf16 == stock behaviour, so a missing
# control file can only ever fall back to stock, never silently to a lossy arm.
export PATH=/opt/venv/bin:$PATH
# OMP left to vLLM (32/worker): the startup firestorm IS the parallel compile. Measured 2026-10-03: capping to 8 made the cold first step 26.67 s vs 9.94 s and cost 8.5% warm (35.84 vs 33.04 s). Fix the compile with the persistent cache below, not by starving it.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400
export PYTHONPATH=/mnt/2king/build/h3/research/2026-10-03-step-profile/port:$PYTHONPATH

# W2/NCCL: documented runtime contract (cu134-bleeding-image-plan.md): the image ships LIL NCCL 2.31.2
# and the sm120 devComm API (ncclDevCommCreate) requires it; without this the lane silently runs
# the system NCCL 2.30.7 and the fused a2a permute fails with "invalid usage". Must be set before
# torch is imported, hence before `exec vllm serve`.
export H3_NCCL_IMAGE=${H3_NCCL_IMAGE:-1}
if [ "$H3_NCCL_IMAGE" = "1" ] && [ -f "/opt/venv/lib/python3.12/site-packages/local_inference_nccl/lib/libnccl.so.2.31.2" ]; then
  export LD_PRELOAD="/opt/venv/lib/python3.12/site-packages/local_inference_nccl/lib/libnccl.so.2.31.2${LD_PRELOAD:+:$LD_PRELOAD}"
  export VLLM_NCCL_SO_PATH="/opt/venv/lib/python3.12/site-packages/local_inference_nccl/lib/libnccl.so.2.31.2"
fi
# W2 permute arm (boot-time: the strategy is built at model load). 0 = NCCL fallback a2a.
export H3_A2A_PERMUTE=${H3_A2A_PERMUTE:-0}

# ---- arm selector: H3_QUANT (mxfp8|hybrid|nvfp4), H3_STEPS (2|4|8),
#      H3_WEIGHTS_SOURCE (local|hf). Defaults: mxfp8, 4, local. ----
source "$(dirname "${BASH_SOURCE[0]}")/h3_lane_env.sh"
if [ "${H3_PRINT:-0}" = "1" ]; then exit 0; fi

# Topology switch (SGLang runs pure sequence parallelism: TP1 + Ulysses, no tensor parallelism).
# Their measured H200 table: Ulysses4 74.38 s E2E vs TP2+Ulysses2 78.33 s, and their docs call TP
# the worst comm axis (per-block all-reduce, every layer). Our own profile agrees: TP all-reduce is
# 918 ms of a 3.525 s step (106 calls, 26%), while per-rank GEMM FLOPs are invariant under TP*USP=4.
# H3_TOPOLOGY=usp4 selects TP1+USP4; anything else keeps today's TP2+USP2.
export H3_TOPOLOGY=${H3_TOPOLOGY:-tp2usp2}
case "$H3_TOPOLOGY" in
  usp4|tp1usp4|usp2x4) H3_TP=1; H3_USP=4; H3_ENC_TP=${H3_ENC_TP:-1} ;;
  tp4|tp4usp1|nousp)     H3_TP=4; H3_USP=1; H3_ENC_TP=${H3_ENC_TP:-4} ;;  # encoder must shard over the 4-rank TP group (validator: "cannot shard the text encoder 2-way")
  tp2usp1|2gpu)         H3_TP=2; H3_USP=1; H3_ENC_TP=${H3_ENC_TP:-2} ;;  # 2 ranks: full sequence per rank, no a2a
  tp1usp2|tp1ag2)       H3_TP=1; H3_USP=2; H3_ENC_TP=${H3_ENC_TP:-1} ;;  # 2 ranks, all heads per rank
  tp2usp2|*|"")         H3_TP=2; H3_USP=2; H3_ENC_TP=${H3_ENC_TP:-2} ;;
esac
echo "[lane] topology=$H3_TOPOLOGY -> tp=$H3_TP usp=$H3_USP text_encoder_tp=$H3_ENC_TP"

# VAE decode compile arm (checkpoint-side levers, all inert unless H3_VAE_COMPILE=1):
#   MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE   - decoder ViT feed-forward
#   MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE - rotary-pos-emb application
# NOTE the sm120 gap: ops/vae/dispatch.py's H3_VAE_OPERATOR_TABLE covers sm90/sm100/sm103 only,
# so on these sm120 cards resolve_h3_vae_operators() returns None and the fused VAE ops were never
# installed - i.e. the compile path does NOT replace any fused kernel on this lane.
export H3_STAGE_TIMING=${H3_STAGE_TIMING:-0}
export H3_VAE_COMPILE=${H3_VAE_COMPILE:-0}
if [ "$H3_VAE_COMPILE" = "1" ]; then
  export MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE=1
  export MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE=1
  export MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE_MODE=${H3_VAE_COMPILE_MODE:-default}
  export MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE_MODE=${H3_VAE_COMPILE_MODE:-default}
  export MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE_FULLGRAPH=${H3_VAE_COMPILE_FULLGRAPH:-0}
fi

# triton_kernels shadowing repair: the NGC base image ships an OLD triton_kernels in
# dist-packages (matmul.py/reduce.py/distributed.py, no matmul_ogs/routing) and vLLM prioritises any
# site-packages-visible triton_kernels over its own complete vendored copy
# (vllm/utils/import_utils.py:64-76), so every process logged
#   ERROR [config.py:29] Failed to import Triton kernels ... No module named 'triton_kernels.matmul_ogs'
# and vLLM silently lost that MoE backend. Give the vendored copy precedence (idempotent, and
# refresh when vLLM's own tree is newer, e.g. after an image rebuild).
VK=/opt/venv/lib/python3.12/site-packages
VSRC=$VK/vllm/third_party/triton_kernels
VDST=$VK/triton_kernels
if [ -d "$VSRC" ] && { [ ! -f "$VDST/matmul_ogs.py" ] || [ "$VSRC/matmul_ogs.py" -nt "$VDST/matmul_ogs.py" ]; }; then
  rm -rf "$VDST" 2>/dev/null
  cp -a "$VSRC" "$VDST" 2>/dev/null && echo "[lane] triton_kernels: vendored copy installed (shadowing fix)"
fi
# Linear arm (2026-10-04): mxfp8 = stock/default, nvfp4 = the W4A4 experiment. The model's single
# construction site selects h3_nvfp4.Nvfp4Linear when this is 1 (minimax_h3_transformer.py alias
# patch). Isolated measurement: 2.13x on the four linears (13.929 -> 6.540 ms per block-set) at
# 3.6x the quantisation error; receipts in profile/sparse-attn-01/fp4/FINDINGS-nvfp4.md. Boot-time
# choice only - the class is picked while the model loads, so an arm change is a reboot.
# quantisation arm (default MXFP8; hybrid NVFP4 and full NVFP4 are options) is set below by
# h3_lane_env.sh from H3_QUANT. Do not set VLLM_OMNI_DIT_* here.
# a2a qkv batching (2026-10-04): the strict-Ulysses q/k/v/gate forward quartet becomes ONE
# collective per DiT block (4 x 50 = the 200 all_to_all calls/step in the step profile).
# Runtime-flippable through the control file so one boot can A/B both arms; 0 == stock.
export H3_A2A_QKV_BATCH_CONTROL=${H3_A2A_QKV_BATCH_CONTROL:-/mnt/2king/build/h3/research/2026-10-03-step-profile/A2A_QKV_BATCH}
export H3_A2A_QKV_BATCH=0
export H3_A2A_WIRE=bf16
export H3_A2A_WIRE_BUFCACHE=${H3_A2A_WIRE_BUFCACHE:-0}
export H3_A2A_WIRE_CONTROL=/mnt/2king/build/h3/research/2026-10-03-step-profile/A2A_WIRE_MODE
export H3_AR_WIRE=bf16
export H3_AR_WIRE_CONTROL=/mnt/2king/build/h3/research/2026-10-03-step-profile/AR_WIRE_MODE
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export H3_FUSE_LORA="$H3_LORA"
export H3_STEP_PROFILE=${H3_STEP_PROFILE:-0}   # default OFF; =1 for a profiled (non-timing) run
export H3_STEP_PROFILE_STEP=${H3_STEP_PROFILE_STEP:-0}
# Persistent compile caches (DS4.1 parity). /mnt/2king is ALREADY a bind mount into this
# container, so pointing the caches under it makes them survive a container rebuild with no
# mount change. Without this every rebuild recompiles from scratch: measured 2026-10-03, the
# live container had no cache mounts and /root/.cache/vllm did not exist at all.
export TRITON_CACHE_DIR=/mnt/2king/cache/h3/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/2king/cache/h3/inductor
export XDG_CACHE_HOME=/mnt/2king/cache/h3/xdg
export VLLM_CACHE_ROOT=/mnt/2king/cache/h3/vllm
export CUDA_CACHE_PATH=/mnt/2king/cache/h3/cuda
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH"
export SOL_ATTN_TAU=1.0
export SOL_ATTN_CORRECTNESS_GATE=${SOL_ATTN_GATE:-0}   # default OFF; set SOL_ATTN_GATE=1 for a verification run
# SOL_ATTN sparse-forward geometry (2026-10-04): REVERTED, see the note below. `SOL_FWD_GROUP` is
# the router's chunk of blocks scored per outer iteration. Isolated (four cards in parallel, clocks
# warmed, autotuner free, one geometry per card) GROUP=64 measured 12.371 ms/call against 13.332 at
# the shipped 32, i.e. -7.2%, with the output differing by 2.441e-04 = 2^-12, one bf16 ulp.
# **In situ that gain does not exist**: with GROUP=64 the profiled step measured _forward_tma at
# 732.8 / 737.5 / 748.4 / 758.5 / 768.5 / 788.4 ms/step (eight samples over two requests) against
# round-8's 745-758 ms at GROUP=32, and the clip ran 2.93 s/it both ways (15.891/16.041 s engine
# against 15.522/15.634). The kernel's in-situ cost (~15.0 ms/call) exceeds the isolated one
# (13.3 ms) even at lower clocks, because the lane's step keeps L2 busy; a 7% isolated win that is
# 0% in situ is the reason this file ships GROUP=32. Re-measure any kernel change on the *profiled
# step* before shipping it. Receipts: profile/sparse-attn-01/ (probe, drivers, geom4/geom5 results),
# runs/2026-10-04T02-51-23-group64, serve_stack.log.
ATTN='{"default": {"backend": "B12X"},
       "per_role": {"self": {"backend": "SOL_ATTN"},
                    "minimax_h3.token_refiner": {"backend": "B12X"}}}'
export H3_REFMOD_PATHS=${H3_REFMOD_PATHS-/home/giga/refmods_ro/minimaxh3_rocco_v2_refmod.safetensors:/home/giga/refmods_ro/minimaxh3_roxy_v2_refmod.safetensors}
export H3_REFMOD_NORMALIZE=1
echo "=== SERVE REF2VA body=SOL_ATTN refiner=B12X AR-wire arm (karmic image) ==="
echo "    torch=$(python -c 'import torch;print(torch.__version__)') cuda=$(python -c 'import torch;print(torch.version.cuda)')"
echo "    a2a control=$(cat "$H3_A2A_WIRE_CONTROL" 2>/dev/null) ar control=$(cat "$H3_AR_WIRE_CONTROL" 2>/dev/null) linear arm=${H3_LINEAR_ARM:-mxfp8} (VLLM_OMNI_DIT_NVFP4=$VLLM_OMNI_DIT_NVFP4) a2a-qkv-batch=$(cat "$H3_A2A_QKV_BATCH_CONTROL" 2>/dev/null)"
exec vllm serve "$MODEL" \
  --omni --task-type ref2va \
  --lora-path "$H3_LORA" \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --enable-sleep-mode \
  --num-gpus $((H3_TP * H3_USP)) --tensor-parallel-size $H3_TP --usp $H3_USP --ring 1 \
  --text-encoder-tp-size $H3_ENC_TP \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  \
  --diffusion-attention-config "$ATTN"
