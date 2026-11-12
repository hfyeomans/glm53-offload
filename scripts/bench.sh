#!/bin/bash
# bench.sh -- measure a running server the way RESULTS.md was measured.  Decode: 24 different prompts, 512 generated tokens each
# (greedy; GLM-5.3 always thinks, effort max, so most of those tokens are reasoning); the figure is the median of the 24.  Prompt: one
# prompt grown in three steps, with 256 tokens generated after each: the prompt rate and the decode rate AT that depth.  The steps are
# 4K target tokens, then 7/16 and 7/8 of the context (256K: 4K / 115K / 229K target, 3.6K / 100K / 201K real; the filler text counts about
# 12 % short of its target).  Python 3, stdlib only.
#   scripts/bench.sh [--url http://127.0.0.1:8080] [--ctx 256k|512k|1m] [--decode-only] [--out bench-<date>.json]
# Without --ctx the context is read from the server (/props).  --decode-only runs the 24 decode prompts and no prompt steps.
# The deep steps take minutes at 512k and tens of minutes at 1m.  It ends with the summary; bench/report.py <file> prints it again.
set -u
URL=http://127.0.0.1:8080; OUT=bench-$(date +%Y%m%d-%H%M).json; PRESET=; DECODE_ONLY=0
while [ $# -gt 0 ]; do case $1 in
  --url) URL=$2; shift 2;;
  --out) OUT=$2; shift 2;;
  --ctx) PRESET=${2,,}; shift 2;;
  --decode-only) DECODE_ONLY=1; shift;;
  -h|--help) sed -n 2,9p "$0"; exit 0;;
  *) echo "unknown option $1 (see --help)"; exit 2;;
esac; done
command -v python3 >/dev/null || { echo "bench.sh: python3 is not on PATH"; exit 2; }
curl -fs "$URL/health" | grep -q ok || { echo "bench.sh: no healthy server at $URL"; exit 2; }
B=$(dirname "$0")/../bench
if [ $DECODE_ONLY = 1 ]; then
  STEPS=(--no-prefill)
  echo "bench.sh: $URL, the 24 decode prompts only, results in $OUT"
else
  if [ -z "$PRESET" ]; then
    CTX=$(curl -fs "$URL/props" | grep -o 'n_ctx":[0-9]*' | head -n 1 | grep -o '[0-9]*$')
    case ${CTX:-0} in
      262144) PRESET=256k;; 524288) PRESET=512k;; 1000000|1048576) PRESET=1m;;
      *) echo "bench.sh: the server reports a context of ${CTX:-?} tokens, not one of the presets; pass --ctx"; exit 2;;
    esac
  fi
  case $PRESET in
    256k) DEPTHS=4096,114688,229376;;
    512k) DEPTHS=4096,229376,458752;;
    1m)   DEPTHS=4096,458752,917504;;
    *) echo "bench.sh: --ctx is one of 256k, 512k, 1m"; exit 2;;
  esac
  STEPS=(--after-tokens 256 --prefill "$DEPTHS")
  echo "bench.sh: $URL, preset $PRESET, prompt steps $DEPTHS target tokens, results in $OUT"
fi
python3 "$B/probe.py" --url "$URL" --model glm-5.3 --out "$OUT" --reps 1 --prompts-file "$B/prompts-24.txt" --timeout 14400 "${STEPS[@]}" \
  || { echo "bench.sh: the run failed, $OUT is not a complete result"; exit 1; }
python3 "$B/report.py" "$OUT"
