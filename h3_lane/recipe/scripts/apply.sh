#!/bin/bash
# apply.sh — install the h3_lane change set into a target site-packages (default: the
# h3kk container's venv, i.e. run this INSIDE the container).
#
#   apply.sh                dry run: report what would change, write nothing
#   apply.sh --apply        write the patches (timestamped .bak per file) + install port/
#   apply.sh --check        py_compile every touched file and report
#   --target DIR            override the site-packages root (default /opt/venv/lib/python3.12/site-packages)
#   --port-dir DIR          where the shim modules go (default: <target>/.. is wrong; see below)
#
# The shims are NOT installed into site-packages: they are imported by bare name and are
# expected on PYTHONPATH (serve_arwire.sh points it at the port dir). So --port-dir
# defaults to the standing research port dir, and only falls back to a copy next to this bundle.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=/opt/venv/lib/python3.12/site-packages
PORTDIR=/mnt/2king/build/h3/research/2026-10-03-step-profile/port
MODE=dry
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) MODE=apply ;;
    --check) MODE=check ;;
    --target) TARGET="$2"; shift ;;
    --port-dir) PORTDIR="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
  shift
done
TS=$(date +%Y%m%d-%H%M%S)

echo "h3_lane apply — mode=$MODE target=$TARGET port=$PORTDIR"
echo

# ---- 1. shims -------------------------------------------------------------------
if [ "$MODE" = "check" ]; then :; else
  echo "== shims (port/) =="
  for f in "$HERE"/port/*.py; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    if [ -f "$PORTDIR/$b" ] && cmp -s "$f" "$PORTDIR/$b"; then
      echo "  same      $b"
    else
      echo "  would install $b"
      [ "$MODE" = "apply" ] && cp "$f" "$PORTDIR/$b"
    fi
  done
  if [ -d "$HERE/port/h3comm" ]; then
    echo "  h3comm/: $([ -d "$PORTDIR/h3comm" ] && echo present || echo 'would install')"
    [ "$MODE" = "apply" ] && cp -r "$HERE/port/h3comm" "$PORTDIR/" 2>/dev/null
  fi
  echo
fi

# ---- 2. patches -----------------------------------------------------------------
echo "== patches =="
FAIL=0
for p in "$HERE"/patches/*.patch; do
  [ -e "$p" ] || continue
  name=$(basename "$p")
  # our patches carry '# <relpath>' as the first line
  rel=$(sed -n '1s|^# ||p' "$p")
  if [ -z "$rel" ] || [ ! -f "$TARGET/$rel" ]; then
    echo "  SKIP  $name  (target file not found: $rel)"
    continue
  fi
  if patch -p1 --dry-run -s -d "$TARGET" < "$p" >/dev/null 2>&1; then
    echo "  clean $name"
    if [ "$MODE" = "apply" ]; then
      cp -n "$TARGET/$rel" "$TARGET/$rel.bak-h3lane-$TS" 2>/dev/null
      patch -p1 -s -d "$TARGET" < "$p" || { echo "    APPLY FAILED"; FAIL=1; }
    fi
  else
    r=$(patch -p1 --dry-run -d "$TARGET" < "$p" 2>&1 | grep -c "FAILED\|Reversed")
    echo "  ${r} hunk(s) do not apply cleanly: $name  (base differs — rebase needed)"
    FAIL=1
  fi
done
echo

# ---- 3. compile check -----------------------------------------------------------
echo "== py_compile =="
for p in "$HERE"/patches/*.patch; do
  rel=$(sed -n '1s|^# ||p' "$p")
  [ -f "$TARGET/$rel" ] || continue
  if python3 -m py_compile "$TARGET/$rel" 2>/dev/null; then echo "  ok   $rel"; else echo "  FAIL $rel"; FAIL=1; fi
done
for f in "$PORTDIR"/*.py; do
  [ -e "$f" ] || continue
  python3 -m py_compile "$f" 2>/dev/null || { echo "  FAIL $(basename "$f")"; FAIL=1; }
done
echo
[ "$MODE" = "dry" ] && echo "dry run only — nothing written. Re-run with --apply."
[ "$FAIL" = "0" ] && echo "RESULT: clean" || echo "RESULT: issues above (see version-skew caveat in README)"
exit $([ "$MODE" = "dry" ] && echo 0 || echo $FAIL)
