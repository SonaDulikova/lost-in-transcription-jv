#!/usr/bin/env bash
# Copy submission_src into the runtime repo, pack, validate, and optionally run in Docker (CPU).
# Usage: scripts/pack.sh [--run] [--small N]
#   --run      run the packed submission in the official image (CPU, internet blocked)
#   --small N  run on only the first N dev clips (default: all 372)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="${LIT_RUNTIME_REPO:-$HOME/repos/lost-in-transcription-runtime}"
RUN=0; SMALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; shift ;;
    --small) SMALL="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

rm -rf "$RUNTIME/submission_src"/*
cp -r "$REPO/submission_src/." "$RUNTIME/submission_src/"
rm -rf "$RUNTIME/submission_src/__pycache__"
cd "$RUNTIME"
rm -f submission/submission.zip
just pack-submission
just check-submission
du -h submission/submission.zip

if [[ "$RUN" == "1" ]]; then
  MOUNT="$RUNTIME/data/data"
  if [[ "$SMALL" != "0" ]]; then
    SMALL_DIR="$RUNTIME/data/data_small"
    rm -rf "$SMALL_DIR"; mkdir -p "$SMALL_DIR/clips"
    head -n $((SMALL + 1)) data/data/submission_format.csv > "$SMALL_DIR/submission_format.csv"
    tail -n +2 "$SMALL_DIR/submission_format.csv" | cut -d, -f1 | while read -r f; do cp "data/data/clips/$f" "$SMALL_DIR/clips/"; done
    MOUNT="$SMALL_DIR"
  fi
  rm -f submission/submission.csv submission/log.txt
  just GPU_ARGS="" MOUNT_DATA="--mount type=bind,source=$MOUNT,target=/code_execution/data,readonly" run
  echo "rows in submission.csv: $(($(wc -l < submission/submission.csv) - 1))"
fi
