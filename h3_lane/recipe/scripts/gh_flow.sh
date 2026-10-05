#!/usr/bin/env bash
# gh_flow.sh — push branches from super and open PRs, all under the fernandaspets account.
#
# Runs on the WORKSTATION (that is where gh is authenticated). Fetches the branch from super over
# ssh (shallow, because the super clones are shallow) and pushes it to the fork over HTTPS, so the
# token never leaves this machine. See PR-FLOW.md for the why.
#
#   gh_flow.sh push <repo> <branch> [fork-branch]
#   gh_flow.sh pr   <repo> <branch> --title "..." --body-file FILE [--ready]
#   gh_flow.sh status
set -euo pipefail

SUPER_SSH=${SUPER_SSH:-super-lan}

repo_super() {
  case "$1" in
    vllm-omni) echo "/home/giga/vllm-omni-build/rebase-main" ;;
    b12x)      echo "/mnt/2king/build/b12x-h3/b12x" ;;
    Sana)      echo "/mnt/2king/sana-sol" ;;
    *) echo "unknown repo: $1" >&2; exit 2 ;;
  esac
}
repo_fork() {
  case "$1" in
    vllm-omni) echo "fernandaspets/vllm-omni" ;;
    b12x)      echo "fernandaspets/b12x" ;;
    Sana)      echo "fernandaspets/Sana" ;;
    *) echo "unknown repo: $1" >&2; exit 2 ;;
  esac
}
repo_upstream() {
  case "$1" in
    vllm-omni) echo "vllm-project/vllm-omni" ;;
    b12x)      echo "local-inference-lab/b12x" ;;
    Sana)      echo "NVlabs/Sana" ;;
    *) echo "unknown repo: $1" >&2; exit 2 ;;
  esac
}

cmd_push() {
  local repo="$1" branch="$2" remote_branch="${3:-$2}"
  local super_path fork
  super_path=$(repo_super "$repo")
  fork=$(repo_fork "$repo")
  local work
  work=$(mktemp -d)
  trap 'rm -rf "$work"' RETURN

  echo "[gh_flow] fetching $repo:$branch from $SUPER_SSH:$super_path"
  git -C "$work" init -q
  git -C "$work" remote add super "ssh://$SUPER_SSH$super_path"

  # Shallow, because the super clones are shallow and cannot serve full history over ssh.
  local depth
  depth=$(( $(timeout 120 ssh -o BatchMode=yes "$SUPER_SSH" "git -C $super_path rev-list --count $branch" 2>/dev/null || echo 20) + 1 ))
  echo "[gh_flow] depth=$depth"
  git -C "$work" fetch -q --depth="$depth" super "$branch"

  echo "[gh_flow] pushing to $fork:$remote_branch"
  git -C "$work" push "https://github.com/$fork.git" "FETCH_HEAD:refs/heads/$remote_branch" 2>&1 | tail -3
  echo "[gh_flow] done: https://github.com/$fork/tree/$remote_branch"
}

cmd_pr() {
  local repo="$1" branch="$2"; shift 2
  local fork upstream
  fork=$(repo_fork "$repo")
  upstream=$(repo_upstream "$repo")
  gh pr create --repo "$upstream" --head "fernandaspets:$branch" "$@"
}

cmd_status() {
  for r in vllm-omni b12x Sana; do
    printf '%-12s fork=%s\n' "$r" "$(gh api "repos/$(repo_fork "$r")" --jq .full_name 2>/dev/null || echo MISSING)"
  done
  echo "--- open PRs from fernandaspets:"
  for r in vllm-omni b12x Sana; do
    gh pr list --repo "$(repo_upstream "$r")" --head "fernandaspets:" --json number,title,isDraft,headRefName \
      --jq ".[] | \"  \($r) #\(.number) draft=\(.isDraft) \(.headRefName): \(.title)\"" 2>/dev/null || true
  done
}

case "${1:-}" in
  push)   shift; cmd_push "$@" ;;
  pr)     shift; cmd_pr "$@" ;;
  status) cmd_status ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
