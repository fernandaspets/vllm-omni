# PROVENANCE — every input, and where it comes from

This file answers "what does this image pull, and from where?" for every byte. Nothing is read from a
local path; there are no private images and no out-of-tree `PYTHONPATH` shims.

## Sources

| # | input | public source | pin | why / notes |
|---|---|---|---|---|
| 1 | **runtime base** | `ghcr.io/local-inference-lab/vllm` | `@sha256:90be19874c57eb31ac9ba20d7d56ae5b39c09e748e8fdfd4e16b79be2e5faab9` (`karmic-kraken-beta-20261004-6f4bbe44d12bb604`) | the LIL CUDA 13.4 runtime. Anonymous pull verified with `docker manifest inspect` on a host with no registry credentials. |
| 2 | **torchaudio** | `nvcr.io/nvidia/pytorch` | `@sha256:33ef5fc15e8937602d64022209cdb2777b32dadf742f41332023d946041b3c14` | the NVIDIA PyTorch image the runtime is derived from. Anonymous pull verified the same way. The LIL runtime dropped torchaudio but the H3 audio VAE needs it (`vae.py: encode_waveform`), and no public wheel matches this torch ABI, so it is copied from this digest-pinned stage. |
| 3 | **vllm-omni** | `https://github.com/fernandaspets/vllm-omni` | branch `h3/features`, SHA recorded in the build log | our fork; upstream PR [vllm-project/vllm-omni#8487](https://github.com/vllm-project/vllm-omni/pull/8487). Contains the B12X backend, the sol_attn backend, the A2A permute, the MiniMax-H3 model changes, the prequant checkpoints, and the lane's wire + quantisation package `vllm_omni/diffusion/h3/`. |
| 4 | **b12x** | `https://github.com/fernandaspets/b12x` | branch `feat/video-block-sparse`, SHA recorded in the build log | our fork; upstream PR [local-inference-lab/b12x#480](https://github.com/local-inference-lab/b12x/pull/480). The SM120/SM121 kernel library. |
| 5 | **sol_attn** | `third_party/sol_attn/` **in this repo** | vendored | third-party from NVlabs/Sana `sol-engine`, Apache-2.0, with its own `THIRD_PARTY_NOTICES.md`. Vendored rather than fetched: the revision the lane ran predates the upstream autotune change (`key=["T"]` → `key=["N"]`), so pinning current upstream would change numerics. |
| 6 | **python dependencies** | PyPI | `requirements.lock` (**82 entries, every one with a SHA-256 hash**) | **only packages the runtime does not ship**, installed with `--require-hashes --no-deps`: every entry is either absent from the base or already present there at exactly this version, so the install cannot move a package the runtime owns. `runtime-dist.json` (the base's 339 distributions) plus `scripts/gate_runtime.py` prove it at build time. See *Build invariants*. |
| 7 | **apt packages** | Ubuntu 24.04 archives | names pinned in the Dockerfile | X11/GL client libraries and `ffmpeg`. `cv2` (opencv-python, a `cosmos-guardrail` dependency) is imported on the worker spawn path and dies without `libxcb.so.1`. |
| 8 | **lane scripts** | `scripts/` **in this repo** | in-tree | arm selector, boot script, weights fetcher, render probe, apply/teardown helpers. |
| 9 | **model weights** | Hugging Face | see `WEIGHTS.md` | deliberately **not** baked into the image. |

## What is deliberately *not* here

- **No `PYTHONPATH` shim.** The lane's wire and quantisation modules used to be loose files on
  `PYTHONPATH`; they now live in the package as `vllm_omni/diffusion/h3/`, so a checkout of the branch
  is everything needed.
- **No model weights.** A 269 GB download does not belong in an image layer. See `WEIGHTS.md`.
- **No identity RefMods.** They are not published; the lane boots and renders without them. See
  `WEIGHTS.md` for what they add and what is lost.

## Build invariants

The image is only as reproducible as the runtime it layers on, so the build asserts three things and
fails rather than shipping an image that lies about itself:

1. **import gate** - `vllm`, `b12x`, `vllm_omni`, the three `vllm_omni.diffusion.h3` modules,
   `sol_attn` and `torchaudio` all import inside the image.
2. **wire gate** - `vllm_omni/diffusion/distributed/comm.py` still imports `a2a_wire` from
   `vllm_omni.diffusion.h3`, i.e. the int8 all-to-all hook survived the install.
3. **runtime gate** - each of the 339 distributions the base shipped is still at its original version
   (`runtime-dist.json` + `scripts/gate_runtime.py`; the single allow-listed change is `b12x`,
   replaced on purpose by the pinned PR branch).

`pip check` is deliberately *not* used as the gate: the bare base fails it (`wandb 0.28.2` requires
`opentelemetry-api>=1.43.0`; the base has 1.40.0) and a `--no-deps` install of `vllm-omni` adds
"requires X, not installed" lines by construction, so it fails for reasons that are not defects.

It did, however, catch a real one. The first revision of `requirements.lock` was a fresh resolution
taken from the working runtime and silently upgraded **38 packages the runtime already owns** -
numba 0.65->0.68 against vllm's `==0.65.0`, numpy 2.1.0->2.5.3 against lmcache's `<=2.2.6` and
mistral-common's `<2.4`, setuptools 80.9->84.0 against vllm's `<81`, transformers 5.15->5.18 against
vllm-omni's `<5.15`, protobuf 6.33.6->7.36.2 against opentelemetry-proto's and grpcio-tools' `<7`,
plus fsspec, packaging, cuda-bindings, scipy, tokenizers, onnx, llvmlite, networkx, filelock,
huggingface-hub and more. `pip check` went from 1 line on the bare base to 21 on that build. Those
entries are now absent from the lock, so pip installs only what is genuinely missing and the runtime
gate makes their return a build failure.

## Verification

```bash
# both bases pull without credentials
docker manifest inspect ghcr.io/local-inference-lab/vllm@sha256:90be19874c57eb31ac9ba20d7d56ae5b39c09e748e8fdfd4e16b79be2e5faab9
docker manifest inspect nvcr.io/nvidia/pytorch@sha256:33ef5fc15e8937602d64022209cdb2777b32dadf742f41332023d946041b3c14

# no local paths anywhere in the build
grep -nE '/mnt/|/home/|local/h3kk' Dockerfile build.sh && echo 'FAIL: local path' || echo 'ok'

# the lock is fully hashed
grep -c -- '--hash=sha256:' requirements.lock     # must equal the entry count

# the lock cannot perturb the runtime: every entry is missing from the base or same version
python3 - <<'EOF'
import json, re
base = {d['name'].lower().replace('_','-'): d['version'] for d in json.load(open('runtime-dist.json'))}
for line in open('requirements.lock'):
    m = re.match(r'^([A-Za-z0-9_.\-]+)==(\S+)', line)
    if m:
        name, ver = m.group(1).lower().replace('_','-'), m.group(2)
        assert name not in base or base[name] == ver, f'{name} would move {base[name]} -> {ver}'
print('lock invariant holds')
EOF

# the build proves its own payload (import gate + wire gate + runtime gate)
bash build.sh h3-repro
```
