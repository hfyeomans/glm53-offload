#!/bin/bash
# kl.sh -- what a serve.sh setting does to the output: llama-perplexity in its KL-divergence mode, run with a preset's exact
# environment and arguments (bench/kl-shim.sh stands in for llama-server), on the wikitext-2 test text: 8 chunks of 4,096 tokens
# in 4-token batches, the batches decode runs, which the expert cache and the route bias serve.  About 14 minutes a run.
#
#   bench/kl.sh --model <first shard> --wikitext <dir with wiki.test.raw> --out <dir> base <label> [serve.sh options]
#   bench/kl.sh --model <first shard> --wikitext <dir with wiki.test.raw> --out <dir> cmp <label> <base label> [serve.sh options]
#
# base writes <out>/<label>.logits (about 5 GB); cmp scores against it.  Either way the log is <out>/<label>.log.  Run a second
# base-like setting as cmp to see the engine's own run-to-run spread; a difference smaller than that says nothing.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); MODEL=; WT=; OUT=; IMAGE=${GLM53_IMAGE:-glm53-offload:latest}
while [ $# -gt 0 ]; do case $1 in
  --model) MODEL=$2; shift 2;; --wikitext) WT=$2; shift 2;; --out) OUT=$2; shift 2;; *) break;;
esac; done
MODE=${1:-}; L=${2:-}; shift 2 || true
[ -f "$MODEL" ] && [ -f "$WT/wiki.test.raw" ] && [ -n "$OUT" ] && [ -n "$L" ] || { sed -n 2,12p "$0"; exit 2; }
case $MODE in
  base) KA="-f /wt/wiki.test.raw --chunks 8 --kl-divergence-base /kl/$L.logits";;
  cmp)  B=$1; shift; [ -f "$OUT/$B.logits" ] || { echo "kl.sh: no $OUT/$B.logits"; exit 2; }
        KA="--kl-divergence-base /kl/$B.logits --kl-divergence";;
  *) sed -n 2,12p "$0"; exit 2;;
esac
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); WT=$(cd "$WT" && pwd)
docker run --rm --init --gpus '"device=0,1"' --ulimit memlock=-1:-1 --shm-size 2g \
  -v "$(cd "$(dirname "$MODEL")" && pwd)":/models:ro -v "$WT":/wt:ro -v "$OUT":/kl \
  -v "$REPO/recipes":/app/recipes:ro -v "$REPO/docker/serve.sh":/app/serve.sh:ro -v "$REPO/bench/kl-shim.sh":/app/llama-server:ro \
  -e KL_ARGS="$KA" "$IMAGE" --model "/models/$(basename "$MODEL")" --ctx 256k "$@" > "$OUT/$L.log" 2>&1
rc=$?; sed -n '/====== Perplexity statistics/,$p' "$OUT/$L.log" | grep -E 'PPL\(Q\)/PPL\(base\)|Mean    KLD|Same top p'; exit $rc
