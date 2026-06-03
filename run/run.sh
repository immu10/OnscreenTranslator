#!/usr/bin/env bash
# Run main.py, prefix every line of stdout/stderr with a timestamp,
# and tee to logs/run-<timestamp>.log.

set -euo pipefail

# Always run from project root regardless of where this script was invoked.
cd "$(dirname "$0")/.."

mkdir -p logs
LOG="logs/run-$(date +%Y%m%d-%H%M%S).log"

# -u  : unbuffered Python output (so logs are real-time, not block-buffered)
# 2>&1: merge stderr into stdout
# while read : prefix each line with HH:MM:SS.mmm
# tee : write to both terminal and log file
python -u main.py 2>&1 | while IFS= read -r line; do
    printf '%s %s\n' "$(date '+%H:%M:%S.%3N')" "$line"
done | tee "$LOG"
