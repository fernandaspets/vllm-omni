#!/bin/bash
# One ref2va request with explicit steps/flow_shift/duration.
#   $1=out  $2=steps  $3=flow_shift  $4=audio_shift  $5=duration seconds (default 10.0)
# Frames align to 17n+5 at 24 fps (5.0 s -> 124 frames, 10.0 s -> 243).
OUT=${1:-out.mp4}; STEPS=${2:-4}; FLOW=${3:-12}; AFLOW=${4:-3.0}; DUR=${5:-10.0}
SEC=${DUR%%.*}   # the seconds field must be an integer string (^[1-9]\d*$)
cd /home/giga/comfyui/outputs/vllmomni
PROMPT=$(cat /home/giga/comfyui/models/solrefs/prompt.txt)
B64=$(base64 -w0 /home/giga/comfyui/models/solrefs/duo_dialogue_32k.wav)
printf '{"audio_url":"data:audio/wav;base64,%s"}' "$B64" > /tmp/audio_ref.json
echo "  duration ${DUR}s steps=$STEPS"
time curl --fail-with-body -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  --form-string "prompt=$PROMPT" \
  -F 'aspect_ratio=16:9' -F 'width=1344' -F 'height=768' -F 'fps=24' -F "seconds=$SEC" \
  -F "flow_shift=$FLOW" -F "num_inference_steps=$STEPS" -F 'seed=20261005' \
  -F 'audio_reference=</tmp/audio_ref.json' \
  -F "extra_params={\"task\":\"ref2va\",\"duration\":$DUR,\"audio_flow_shift\":$AFLOW}" \
  -o "$OUT"
echo "REQ-EXIT=$?"; ls -la "$OUT" 2>/dev/null | awk '{printf "  clip bytes: %s\n", $5}'
