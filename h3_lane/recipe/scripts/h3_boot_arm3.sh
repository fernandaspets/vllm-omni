#!/bin/bash
# Boot the H3 lane with a chosen LINEAR ARM.
#   ARM=mxfp8|nvfp4  PROFILE=0|1  [LOG=/path] bash h3_boot_arm3.sh
#
# Fixes over h3_boot_arm2.sh, both of which cost a boot:
#  1. the teardown runs INSIDE h3kk via teardown_h3.sh (the lane's processes are root's; a host-side
#     kill is silently permission-denied), and it keys on the H3 model path so it cannot touch the
#     qwen38 vl service or prod containers;
#  2. the boot streams to /proc/1/fd/1 as well as the log file
#     (`... 2>&1 | tee -a $LOG /proc/1/fd/1 > /dev/null`) so `docker logs -f h3kk` is live - PID 1 is
#     `sleep infinity`, so a plain `> $LOG` redirect leaves docker logs frozen on an earlier boot.
D=/mnt/2king/build/h3/research/2026-10-03-step-profile
ARM=${ARM:-mxfp8}
PROFILE=${PROFILE:-0}
H3_A2A_PERMUTE=${H3_A2A_PERMUTE:-0}
H3_NCCL_IMAGE=${H3_NCCL_IMAGE:-1}
H3_VAE_COMPILE=${H3_VAE_COMPILE:-0}
H3_STAGE_TIMING=${H3_STAGE_TIMING:-0}
H3_TOPOLOGY=${H3_TOPOLOGY:-tp2usp2}
H3_BOOT_WARMUP=${H3_BOOT_WARMUP:-1}
H3_LORA=${H3_LORA:-}
SUFFIX=""; [ "$PROFILE" = "1" ] && SUFFIX="_prof"
LOG=${LOG:-$D/serve_arm_${ARM}${SUFFIX}.log}

: > /tmp/h3_boot_warmup.log   # stale-warmup guard: the runner waits on this file
docker exec h3kk bash "$D/teardown_h3.sh"

for i in $(seq 1 12); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8000/health 2>/dev/null)
  echo "  pre-boot check $i: health=${code:-000} lane GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed -n '3,6p' | tr '\n' ' ')"
  [ "${code:-000}" = "000" ] && break
  if [ "$i" = "12" ]; then
    echo "REFUSING: port 8000 still answers ${code} after teardown; refusing to start a doomed boot."
    docker exec h3kk bash -lc "ps -eo pid,args | grep -E '[M]iniMax-H3-ref2va|[v]LLM-Omni::Diff' | cut -c1-120"
    exit 1
  fi
  sleep 5
done

echo "=== boot arm=$ARM profile=$PROFILE $(date +%H:%M:%S) -> $LOG (also /proc/1/fd/1) ==="
: > "$LOG"
docker exec -d h3kk bash -c "cd $D && H3_A2A_PERMUTE=$H3_A2A_PERMUTE H3_A2A_WIRE_BUFCACHE=${H3_A2A_WIRE_BUFCACHE:-0} H3_NCCL_IMAGE=$H3_NCCL_IMAGE H3_REFMOD_PATHS="${H3_REFMOD_PATHS:-}" H3_VAE_COMPILE=$H3_VAE_COMPILE H3_STAGE_TIMING=$H3_STAGE_TIMING H3_TOPOLOGY=$H3_TOPOLOGY H3_LORA=$H3_LORA H3_BOOT_WARMUP=$H3_BOOT_WARMUP H3_LINEAR_ARM=$ARM H3_STEP_PROFILE=$PROFILE bash serve_arwire.sh 2>&1 | tee -a $LOG /proc/1/fd/1 > /dev/null"
echo "launched $(date +%H:%M:%S)"

for i in $(seq 1 100); do
  sleep 15
  code=$(docker exec h3kk curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/health 2>/dev/null)
  last=$(tail -1 "$LOG" 2>/dev/null | tr -d '\r' | cut -c1-110)
  echo "[$i] $(date +%H:%M:%S) http=${code:-none} | $last"
  if [ "$code" = "200" ]; then
    echo "READY $(date +%H:%M:%S)"
    echo "--- arm evidence from the lane log:"
    grep -a "linear arm=\|h3_nvfp4" "$LOG" 2>/dev/null | head -10
    if [ "${H3_BOOT_WARMUP:-0}" = "1" ]; then
      : > /tmp/h3_boot_warmup.log
      setsid nohup bash /mnt/2king/build/w2-a2a-permute/warmup_h3.sh >> /tmp/h3_boot_warmup.log 2>&1 < /dev/null &
      echo "[lane] boot warmup launched (served-shape canary)"
    fi
    exit 0
  fi
done
echo "TIMEOUT after 25 min"
exit 1

