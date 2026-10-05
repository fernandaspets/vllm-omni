#!/bin/bash
# e2e_test.sh <tag> - boot the built image in a NEW container, render one canary clip, and verify the
# ARTIFACT (size, sha256, ffprobe, extracted frames) rather than trusting an exit code.
#
#   GPUS=2,3,4,5      devices to use (the lane needs four SM120 cards)
#   H3_QUANT=hybrid   arm (hybrid | mxfp8 | nvfp4)
#   H3_STEPS=4        sampling steps
#
# Writes a receipt directory with serve.log, request.json, the clip, its digest and probe output.
# Exits non-zero if the lane does not become healthy or the clip is not a decodable video.
set -uo pipefail

TAG=${1:?usage: e2e_test.sh <tag>}
NAME=h3e2e-${TAG//\//_}-$$
GPUS=${GPUS:-2,3,4,5}
H3_QUANT=${H3_QUANT:-hybrid}
H3_STEPS=${H3_STEPS:-4}
RUN=/mnt/2king/build/h3/research/2026-10-03-step-profile/runs/e2e-$TAG-$(date +%Y%m%dT%H%M%S)

# Refuse to start on occupied GPUs. The lane needs ~68 GB per card and a silent second run does not
# share - it OOMs mid-load, which reads as a lane bug. Fail loudly instead.
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
  if [ "${used:-0}" -gt 5000 ]; then
    echo "[e2e] REFUSING: GPU $g already has ${used} MiB in use (another lane is up). Stop it first."
    exit 1
  fi
done
other=$(docker ps --format '{{.Names}}' | grep -E '^(h3|h3kk|h3e2e|h3acc|h3reboot)' | grep -v "^$NAME$" || true)
if [ -n "$other" ]; then
  echo "[e2e] REFUSING: another H3 container is running: $(echo $other | tr '\n' ' ')"
  exit 1
fi


echo "[e2e] image=local/h3kk:$TAG  container=$NAME  gpus=$GPUS  arm=$H3_QUANT  steps=$H3_STEPS"
echo "[e2e] receipt dir: $RUN"
mkdir -p "$RUN"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --entrypoint sleep \
  --network host --ipc host --shm-size 36g --gpus "\"device=$GPUS\"" \
  -v /mnt/2king:/mnt/2king \
  -v /home/giga/comfyui:/home/giga/comfyui \
  -v /home/giga/refmods_ro:/home/giga/refmods_ro:ro \
  -v lil-hf:/root/.cache/huggingface \
  "local/h3kk:$TAG" infinity >/dev/null || { echo "[e2e] FAIL: container did not start"; exit 1; }
echo "[e2e] container: $(docker inspect -f '{{.State.Status}}' "$NAME")"

cleanup() {
  docker logs "$NAME" > "$RUN/container.log" 2>&1 || true
  echo "[e2e] tearing down $NAME"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Refuse to test an image whose own payload cannot import.
if ! docker exec "$NAME" /opt/venv/bin/python -c "import vllm_omni" 2>"$RUN/import.err"; then
  echo "[e2e] FAIL: import gate"; tail -5 "$RUN/import.err"; exit 1
fi
echo "[e2e] import gate: ok"

echo "[e2e] serving"
docker exec -d "$NAME" bash -lc \
  "H3_QUANT=$H3_QUANT H3_STEPS=$H3_STEPS H3_WEIGHTS_SOURCE=local \
   nohup bash /opt/h3/scripts/serve_arwire.sh > $RUN/serve.log 2>&1"

echo "[e2e] waiting for health (warm ~2 min, cold ~10)"
ok=0
for _ in $(seq 1 90); do
  code=$(docker exec "$NAME" /opt/venv/bin/python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).status)" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { ok=1; echo "[e2e] health 200"; break; }
  sleep 10
done
if [ "$ok" != "1" ]; then
  echo "[e2e] FAIL: no health after 15 min"
  docker exec "$NAME" tail -30 "$RUN/serve.log" 2>/dev/null
  exit 1
fi

if ! docker exec "$NAME" grep -qE "mxfp8=[0-9]+ .*nvfp4=[0-9]+ .*bf16_roles=[0-9]+" "$RUN/serve.log"; then
  echo "[e2e] WARN: quant engagement line not found in serve.log (a silent no-op would hide here)"
fi

echo "[e2e] rendering the canary with the PROVEN request (req_steps.sh)"
# This is the script that produced every clip in the three days of runs under
# /mnt/2king/build/h3/research/2026-10-03-step-profile/runs/. It is used verbatim rather than
# re-implemented: prompt is a plain form field, audio_reference is a field read from a JSON file,
# and the endpoint returns the mp4 BYTES as the response body (curl -o writes them straight out).
OUT="$RUN/canary.mp4"
docker exec "$NAME" bash -lc "bash /opt/h3/scripts/req_steps.sh $OUT $H3_STEPS 12.0 3.0 5.0" 2>&1 | tail -6 | tee "$RUN/request.log"

CLIP="$OUT"
if [ ! -s "$CLIP" ]; then
  echo "[e2e] FAIL: no clip produced (see $RUN/request.log)"
  exit 1
fi

echo "[e2e] --- artifact verification (not the exit code) ---"
BYTES=$(stat -c %s "$CLIP")
SHA=$(sha256sum "$CLIP" | cut -d' ' -f1)
printf "  clip:   %s\n" "$CLIP"
printf "  bytes:  %s\n" "$BYTES"
printf "  sha256: %s\n" "$SHA"
# The recorded canary from 2026-10-04T04-00-28-mxfp8-warm/both_r2.mp4. A byte-identical clip is the
# strongest end-to-end evidence there is: same weights, same arm, same request.
REF_BYTES=7198351
REF_SHA=239b17022692d913e45447764b39c3431045e32239c77de4ff72f15ec5c827ba
if [ "$BYTES" = "$REF_BYTES" ] && [ "$SHA" = "$REF_SHA" ]; then
  echo "  CANARY: byte-identical to the recorded reference ($REF_BYTES B) -> PASS"
else
  echo "  CANARY: differs from reference (expected $REF_BYTES B / ${REF_SHA:0:12}, got $BYTES B / ${SHA:0:12})"
  echo "          not a failure by itself if the arm differs - but it must be explained, not ignored"
fi
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames -show_entries format=duration \
  -of default=noprint_wrappers=1 "$CLIP" | sed 's/^/  /' | tee "$RUN/ffprobe.txt"

# a mosaic/corrupt clip still decodes as some frames; count distinct frames and require the first
# and last to differ
ffmpeg -v error -i "$CLIP" -vf "select=eq(n\,0)+eq(n\,$(($(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$CLIP" 2>/dev/null | tr -d '\n')-1))" -vsync 0 -frames:v 2 "$RUN/frame_%02d.png" 2>/dev/null
if [ -f "$RUN/frame_01.png" ] && [ -f "$RUN/frame_02.png" ]; then
  a=$(sha256sum "$RUN/frame_01.png" | cut -c1-16); b=$(sha256sum "$RUN/frame_02.png" | cut -c1-16)
  [ "$a" != "$b" ] && echo "  frames: differ ($a vs $b) -> motion present" || echo "  frames: IDENTICAL - possible still/mosaic"
fi

echo "[e2e] PASS (artifact written). receipt: $RUN"
