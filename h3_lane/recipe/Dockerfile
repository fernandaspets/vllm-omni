# syntax=docker/dockerfile:1.7
#
# MiniMax-H3 ref2va lane - reproducible build.
#
# Every input below is a PUBLIC source pinned by revision or digest. There are no local paths, no
# private images and no out-of-tree PYTHONPATH shims: the application code is cloned from GitHub at
# the revisions named in ARG, and everything else is a digest-pinned image or a hashed wheel.
#
# Provenance for each input, and what it is, is in PROVENANCE.md. Model weights are NOT in this
# image; see WEIGHTS.md.

ARG LIL_RUNTIME=ghcr.io/local-inference-lab/vllm@sha256:90be19874c57eb31ac9ba20d7d56ae5b39c09e748e8fdfd4e16b79be2e5faab9

# ---- stage 2: the lane -------------------------------------------------------------------------
FROM ${LIL_RUNTIME}

ARG VLLM_OMNI_REPO=https://github.com/fernandaspets/vllm-omni
ARG VLLM_OMNI_SHA=PLACEHOLDER_VLLM
ARG B12X_REPO=https://github.com/fernandaspets/b12x
ARG B12X_SHA=PLACEHOLDER_B12X

LABEL org.opencontainers.image.title="MiniMax-H3 ref2va lane" \
      org.opencontainers.image.base.name="${LIL_RUNTIME}" \
      org.opencontainers.image.revision="vllm-omni@${VLLM_OMNI_SHA} b12x@${B12X_SHA}"

USER root

# 0. System libraries the runtime does not carry. cv2 (opencv-python, a cosmos-guardrail
#    dependency) is imported on the worker spawn path and needs these, or the workers die with
#    `ImportError: libxcb.so.1`.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libxcb1 libx11-6 libxext6 libxrender1 libsm6 libice6 libxau6 libxdmcp6 \
      libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 libglvnd0 \
      libxfixes3 libxi6 libxrandr2 libxcursor1 libxcomposite1 libxdamage1 libxinerama1 \
      libglib2.0-0 libgomp1 libsdl2-2.0-0 ocl-icd-libopencl1 ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# 1. torchaudio, from the public PyTorch nightly index.
#    The LIL runtime does not ship it and the H3 audio VAE imports it (vae.py). This wheel is the
#    exact build the lane ran (2.11.0.dev20261001+cu134), published as a cp310-abi3 wheel that is
#    valid on this 3.12 runtime. Pinned by URL + SHA-256; --no-deps because torch comes from the
#    runtime.
ARG TORCHAUDIO_URL="https://download-r2.pytorch.org/whl/nightly/cu134/torchaudio-2.11.0.dev20261001%2Bcu134-cp310-abi3-manylinux_2_28_x86_64.whl"
ARG TORCHAUDIO_SHA256="46ccf9f6e814972b4a4a85eaadab38f0c049dac05e794624bcffdbdf7e5fb49e"
RUN /opt/venv/bin/python -m pip install --no-deps "https://download-r2.pytorch.org/whl/nightly/cu134/torchaudio-2.11.0.dev20261001%2Bcu134-cp310-abi3-manylinux_2_28_x86_64.whl#sha256=46ccf9f6e814972b4a4a85eaadab38f0c049dac05e794624bcffdbdf7e5fb49e" \
 && /opt/venv/bin/python -c "import torchaudio, importlib.metadata as m; print('torchaudio', m.version('torchaudio'))"

# 2. vllm-omni, cloned at the pinned revision. --no-deps on purpose: the runtime provides vllm and
#    the rest; letting pip resolve here would silently replace them.
RUN git clone --filter=blob:none "${VLLM_OMNI_REPO}" /src/vllm-omni \
 && git -C /src/vllm-omni checkout --detach "${VLLM_OMNI_SHA}" \
 && SETUPTOOLS_SCM_PRETEND_VERSION="0.1.dev0+g$(git -C /src/vllm-omni rev-parse --short HEAD)" \
    /opt/venv/bin/python -m pip install --no-deps --no-build-isolation /src/vllm-omni \
 && rm -rf /src/vllm-omni

# 3. b12x, cloned at the pinned revision.
RUN git clone --filter=blob:none "${B12X_REPO}" /src/b12x \
 && git -C /src/b12x checkout --detach "${B12X_SHA}" \
 && /opt/venv/bin/python -m pip install --no-deps --no-build-isolation /src/b12x \
 && rm -rf /src/b12x

# 3b. triton_kernels precedence, baked in rather than fixed at boot. vLLM vendors a complete copy
#     under vllm/third_party/ (it still has matmul_ogs.py); the runtime's dist-packages copy is the
#     post-3.7 tree (matmul.py) and does not. vLLM's importer prefers whatever is importable from
#     site-packages, so without this every process logs
#       ERROR [config.py:29] Failed to import Triton kernels ... No module named 'triton_kernels.matmul_ogs'
#     and silently loses that MoE backend. This is a namespace precedence, not a downgrade: the
#     copy is vLLM's own, taken from the installed vllm.
RUN test -f /opt/venv/lib/python3.12/site-packages/vllm/third_party/triton_kernels/matmul_ogs.py \
 && rm -rf /opt/venv/lib/python3.12/site-packages/triton_kernels \
 && cp -a /opt/venv/lib/python3.12/site-packages/vllm/third_party/triton_kernels \
          /opt/venv/lib/python3.12/site-packages/triton_kernels \
 && /opt/venv/bin/python -c "import triton_kernels.matmul_ogs as m; print('triton_kernels precedence: ok', m.__name__)"

# 4. the packages this runtime does not ship, from the hashed lock.
#    --no-deps is not an optimisation here, it is the contract: the lock names ONLY packages the
#    runtime lacks, so pip must never "resolve" its way to newer versions of the ones it owns.
#    Letting it do so is how the previous revision pulled numba 0.68 over vllm's pinned 0.65, numpy
#    2.5 over lmcache's 2.2.6 ceiling, setuptools 84 over vllm's <81, transformers 5.18 over
#    vllm-omni's <5.15 and protobuf 7 over opentelemetry-proto's <7 -- 38 runtime packages moved, and
#    the build gate then failed on `pip check`.
COPY requirements.lock /tmp/requirements.lock
RUN /opt/venv/bin/python -m pip install --require-hashes --no-deps -r /tmp/requirements.lock \
 && rm -f /tmp/requirements.lock

# 5. sol_attn. Vendored third-party from NVlabs/Sana (sol-engine), Apache-2.0, with its own
#    THIRD_PARTY_NOTICES.md. Vendored rather than fetched because the revision the lane ran predates
#    the upstream autotune change; see PROVENANCE.md.
COPY third_party/sol_attn /opt/venv/lib/python3.12/site-packages/sol_attn

# 6. the lane's own scripts.
COPY scripts /opt/h3/scripts
RUN chmod +x /opt/h3/scripts/*.sh /opt/h3/scripts/*.py 2>/dev/null || true

# 7. BUILD GATE. A build that cannot import its payload, that lost the wire hook the speed work
#    consists of, or that moved a package the runtime shipped, is a failed build - not something to
#    discover later in a boot test. The last check replaces `pip check`, which cannot serve as a gate
#    here: the bare base already fails it (wandb wants opentelemetry-api>=1.43, base has 1.40), and a
#    --no-deps install of vllm-omni adds "requires X, not installed" lines by construction. The
#    invariant that must actually hold is that the runtime's version set survives unchanged.
COPY runtime-dist.json /tmp/runtime-dist.json
RUN /opt/venv/bin/python -c "import importlib; [importlib.import_module(m) for m in ('vllm','b12x','vllm_omni','vllm_omni.diffusion.h3.a2a_wire','vllm_omni.diffusion.h3.quant_policy','vllm_omni.diffusion.h3.nvfp4','sol_attn','torchaudio')] ; import sol_attn.interface as _i ; assert (12,0) not in _i._CUTE_BACKENDS, 'SM120 must fall back to the Triton backend: (12,0) is in _CUTE_BACKENDS but sol_attn ships no sm120 kernel' ; print('import gate: ok (SM120 -> triton)')" \
 && grep -q "from vllm_omni.diffusion.h3 import a2a_wire" /opt/venv/lib/python3.12/site-packages/vllm_omni/diffusion/distributed/comm.py \
 && echo "wire gate: ok" \
 && /opt/venv/bin/python /opt/h3/scripts/gate_runtime.py /tmp/runtime-dist.json \
 && rm -f /tmp/runtime-dist.json

ENTRYPOINT ["/usr/local/bin/lil-entrypoint"]
CMD []
