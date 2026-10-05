#!/bin/bash
# Build the lane image from public sources only.
#   build.sh [tag]
# Every input is pinned below; nothing is read from this machine.
set -euo pipefail
cd "$(dirname "$0")"
TAG=${1:-h3-repro}
VLLM_OMNI_SHA=$(git ls-remote https://github.com/fernandaspets/vllm-omni refs/heads/h3/features | cut -f1)
B12X_SHA=$(git ls-remote https://github.com/fernandaspets/b12x refs/heads/feat/video-block-sparse | cut -f1)
echo "[build] vllm-omni $VLLM_OMNI_SHA"
echo "[build] b12x      $B12X_SHA"
docker build \
  --build-arg "VLLM_OMNI_SHA=$VLLM_OMNI_SHA" \
  --build-arg "B12X_SHA=$B12X_SHA" \
  -t "local/h3kk:$TAG" .
docker run --rm --entrypoint /opt/venv/bin/python "local/h3kk:$TAG" -c "
import importlib.metadata as m
for p in ('vllm','b12x','vllm_omni'):
    print('   ', p, m.version(p))
"
