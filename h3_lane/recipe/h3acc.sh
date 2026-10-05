#!/bin/bash
# Boot the recipe image in a NEW persistent container, render one canary, leave the container up.
set -uo pipefail
TAG=${TAG:-h3-repro-v2}; NAME=${NAME:-h3acc}
RUN=/mnt/2king/build/h3/research/2026-10-03-step-profile/runs/acc-$TAG-$(date +%Y%m%dT%H%M%S)
mkdir -p "$RUN"; echo "[acc] image=local/h3kk:$TAG container=$NAME"; echo "[acc] run dir $RUN"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --entrypoint sleep --network host --ipc host --shm-size 36g \
  --gpus "\"device=2,3,4,5\"" -v /mnt/2king:/mnt/2king -v /home/giga/comfyui:/home/giga/comfyui \
  -v /home/giga/refmods_ro:/home/giga/refmods_ro:ro -v lil-hf:/root/.cache/huggingface \
  "local/h3kk:$TAG" infinity >/dev/null || { echo "[acc] FAIL: start"; exit 1; }
echo "[acc] container up: $(docker inspect -f '{{.State.Status}}' "$NAME")"
docker exec -d "$NAME" bash -lc "H3_QUANT=${H3_QUANT:-hybrid} H3_STEPS=${H3_STEPS:-4} H3_WEIGHTS_SOURCE=local nohup bash /opt/h3/scripts/serve_arwire.sh > $RUN/serve.log 2>&1"
echo "[acc] serving; waiting for health"
code=000
for i in $(seq 1 90); do
  code=$(docker exec "$NAME" /opt/venv/bin/python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).status)" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "[acc] health 200 after $((i*10))s"; break; }
  sleep 10
done
[ "$code" = "200" ] || { echo "[acc] FAIL: no health; last log:"; tail -20 "$RUN/serve.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-160; exit 1; }
echo "[acc] engagement proof:"; grep -aoE "mxfp8=[0-9]+ \([0-9]+ shapes?\)|nvfp4=[0-9]+ \([0-9]+ shapes?\)|bf16_roles=[0-9]+" "$RUN/serve.log" | tail -3
echo "[acc] rendering"
docker exec "$NAME" env STEPS="${H3_STEPS:-4}" OUT_DIR="$RUN" REFS="${REFS-}" \
  /opt/venv/bin/python /opt/h3/scripts/h3_render_request.py | tee "$RUN/request.json"
CLIP=$(find "$RUN" -name '*.mp4' 2>/dev/null | head -1)
if [ -n "$CLIP" ]; then
  echo "[acc] --- artifact ---"; stat -c '  bytes:  %s' "$CLIP"; sha256sum "$CLIP" | sed 's/^/  sha256: /'
  ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames -show_entries format=duration -of default=noprint_wrappers=1 "$CLIP" | sed 's/^/  /'
else echo "[acc] FAIL: no clip produced"; fi
echo "[acc] container $NAME left running for iteration"
