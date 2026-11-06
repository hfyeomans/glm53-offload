#!/bin/bash
# bench.sh -- measure a running server the way RESULTS.md was measured: 3 x 512-token story and code answers (greedy; GLM-5.3 always thinks, effort max), then
# prompts of 4K / 28K / 57K target tokens (3.6K / 25K / 50K real) with 256 tokens generated after each.  Python 3, stdlib only.
#   scripts/bench.sh [--url http://127.0.0.1:8080] [--out bench-<date>.json]
set -u
URL=http://127.0.0.1:8080; OUT=bench-$(date +%Y%m%d-%H%M).json
while [ $# -gt 0 ]; do case $1 in --url) URL=$2; shift 2;; --out) OUT=$2; shift 2;; *) echo "unknown option $1"; exit 2;; esac; done
curl -fs "$URL/health" | grep -q ok || { echo "bench.sh: no healthy server at $URL"; exit 2; }
exec python3 "$(dirname "$0")/../bench/probe.py" --url "$URL" --model glm-5.3 --out "$OUT" --reps 3 --after-tokens 256 --prefill 4096,28672,57344
