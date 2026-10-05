# INVENTORY — the H3 lane's proven state (evidence base: 2026-10-03 → 2026-10-04)

Written after being told, correctly, that the recipe and PRs must be derived from a deep study of the
past days' work rather than rebuilt from first principles. **Today's (10-05) recipe/e2e work is
excluded as evidence — it is the mess, not the reference.**

Source of truth: `/mnt/2king/build/h3/research/2026-10-03-step-profile/` on super.

## 1. What the standing lane actually is

| component | value | how it is anchored |
|---|---|---|
| image | `local/h3kk:base` = `sha256:54a9e0b40dfd1b70cc8cd560f04d7d51f440a9cdbceb49ae4bd16cddf94d2ddb` | recorded `container_image` in **every** 10-04 `env.json` |
| launcher | `serve_arwire.sh` @ `93cbdd0caeba…` (9930 B) | the **last** proven run `2026-10-04T08-57-09-boot-warmup` |
| request | `req_steps.sh` @ `d340fd5a3ab1…` (1195 B) | **unchanged across all of 10-04** |
| runtime | `vllm 0.1.dev21986+g4a379ed42.cu134` (jovian tooling, `RUNTIME_SOURCE_COMMIT=6c0e9843`) | `env.json` + container `pip freeze` |
| arm | `H3_QUANT=hybrid H3_STEPS=4 H3_WEIGHTS_SOURCE=local` | `RECEIPT-branch-reproduction.md` |
| identity | **RefMods via `H3_REFMOD_PATHS`** — *not* raw reference images | `RECEIPT-branch-reproduction.md` |

`env.json` is the mechanism to trust here: each run records the **sha256 of every script and port
module** it used, plus the image digest and a GPU snapshot. That is a real fingerprint, and it is why
"which launcher was standing?" is answerable to the byte.

## 2. The launcher's evolution (14 distinct versions on 10-04)

The launcher was revised all day. Last state wins:

```
04:00  46c8d5e90cb9  5267 B   (bak-qkvbatch-20261004-043141)   <- the mxfp8-warm receipts
06:36  e9e0db98247b  9455 B   (bak-tp4-20261004-070204)
07:07  c4bbdd292e8a  9626 B   (bak-2gpu-20261004-071140)
07:22  d98b3ca016cb  9862 B   (bak-bufcache-20261004-073045)
07:33  e1c24b763b8e  9887 B   (bak-refmod-20261004-082907)
08:40  0e5a8210eb08  9931 B   (bak-emptydash)
08:57  93cbdd0caeba  9930 B   <-- LAST PROVEN, still on disk as serve_arwire.sh
```

The recipe as of today ships `252a82725380` (9978 B) — a **different, never-run** launcher. That is a
defect in the recipe, not a refinement of it.

## 3. Recorded canaries (the acceptance targets)

| run | arm | bytes | sha256 (prefix) | engine |
|---|---|---|---|---|
| `04-00-28-mxfp8-warm/both_r2.mp4` | mxfp8 + int8 wire | **7,198,351** | `239b17022692` | 15.786 s |
| standing receipt (hybrid + refmods) | hybrid | **7,896,537** | `df53989f…` | 13.482 s |
| `RECEIPT-branch-reproduction.md` | hybrid, branch `843e20a` | 9,190,983 | `87c8f3a5…` | — (wall 22.372 s @ 260 W) |

Format of all: h264 1344x768, **124 frames, 5.1667 s**, aac 5.175 s. Power matters: the receipts ran
at 480 W, the branch reproduction at 260 W, so wall times are not comparable across them.

## 4. Known incompatibilities (documented before today)

1. **Newest LIL runtime breaks the render.** `RECEIPT-branch-reproduction.md`, verbatim: on
   `kraken-c7f9993f` (vllm `0.1.dev22032`) the branch *boots and engages the hooks*, but the render
   fails with **`mat2 must be a matrix, got 1-D tensor`**. Cause is logged by our own patch: the
   newer upstream serves ModelOpt linears through the generic `ModelOptLinearMethod`, so the NVFP4
   quant path is skipped and the 1-D scale reaches the matmul. **Open. This is the blocker for
   "newest + works".**
2. **`triton_kernels.matmul_ogs` is gone in Triton 3.8.0** (renamed to `matmul`; the rename landed in
   3.7.0). The importer is **vllm's MXFP4/MoE oracle path** (`fused_moe/oracle/mxfp4.py`,
   `fused_moe/config.py`, `gpt_oss_triton_kernels_moe.py`, `quantization/utils/mxfp4_utils.py`) —
   guarded, caught, and unused by an H3 diffusion lane. The standing container worked around it by
   vendoring an old-API `triton_kernels` **inside the venv** (present in `workcode` only; *not* in
   either base image). Do **not** pin old triton; fix the call sites upstream.
3. NCCL: env says `2.30.7+cuda13.3` in both bases, while `local-inference-nccl-cu134==2.31.2+lil…`
   is the installed package in both. The "parity gap" I claimed earlier was wrong.

## 5. What the recipe got wrong (all found the hard way, none of it necessary)

| defect | truth from the evidence |
|---|---|
| 120-package lock resolved from PyPI | the lane is base + **24 packages**; the lock must never move runtime pins |
| `pip check` as the build gate | the pristine base already fails it; use the runtime-dist fingerprint (`gate_runtime.py`) |
| re-invented `h3_render_request.py` | the proven `req_steps.sh` exists and is unchanged all day; it also shows the response **body is the mp4** |
| `prompt` sent as a file part | `prompt` is a plain form field → the 400 |
| raw reference images | the standing method is RefMods; `req_steps.sh` sends **no** `input_references` |
| unproven launcher shipped | must be `93cbdd0caeba` |

## 6. Consequence for the Dockerfile and the PRs

- The recipe must ship the **proven** launcher and request, hash-pinned, and assert those hashes at
  build time against the recorded ones. `env.json` gives the expected values.
- On the newest runtime the deliverable is **not** "builds and boots" — it is either
  (a) the ModelOpt/NVFP4 path reconciled with newer upstream, or (b) an explicit, documented runtime
  pin. Claiming AC4 (canary render) on karmic today would be false.
- PR #8487 currently carries the lane's code with a vendored-shim history; any claim in it about the
  newest runtime must match `RECEIPT-branch-reproduction.md`, not the newer optimistic rewrites.
