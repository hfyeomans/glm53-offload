#!/bin/bash
# serve.sh -- the image's entrypoint: checks the machine, picks the recipe for the GPUs it was given (one or two RTX PRO 6000) and the
# context asked for, sizes the MoE cache and starts llama-server on port 8080 (OpenAI-compatible: /v1/chat/completions).
#
#   serve.sh --model /models/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf [options] [-- extra llama-server args]
#     --ctx 256k           context preset: 256k (default), 512k or 1m
#     --route-bias V       0 (default): the model's own routing.  0.03: the measured speed option; it changes the output (README.md)
#     --cache-mib N        MoE cache size in MiB over all GPUs (default: the preset's measured size, less when a card is not whole)
#     --threads N          CPU threads (default: the physical cores)
#     --alias NAME         model name the API reports (default glm-5.3)
#     --force              start even if a check fails (RAM, GPU model, peer-to-peer, a bias above 0.03, a card with too little free)
#     --dry-run            print the command and the environment, start nothing
set -u
MODEL=; PRESET=256k; BIAS=; CACHE=; THREADS=; ALIAS=glm-5.3; FORCE=0; DRY=0; PASS=()
while [ $# -gt 0 ]; do case $1 in
  --model) MODEL=$2; shift 2;;
  --ctx) PRESET=${2,,}; shift 2;;
  --route-bias) BIAS=$2; shift 2;;
  --cache-mib) CACHE=$2; shift 2;;
  --threads) THREADS=$2; shift 2;;
  --alias) ALIAS=$2; shift 2;;
  --force) FORCE=1; shift;;
  --dry-run) DRY=1; shift;;
  --) shift; PASS=("$@"); break;;
  -h|--help) sed -n 2,12p "$0"; exit 0;;
  *) echo "unknown option $1 (see --help)"; exit 2;;
esac; done

say(){ echo "[serve] $*"; }
fail(){ echo "[serve] ERROR: $*"; if [ $FORCE = 1 ]; then echo "[serve] --force: continuing"; else exit 1; fi; }

[ -n "$MODEL" ] || { echo "[serve] --model is required (the first shard, e.g. /models/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf)"; exit 2; }
[ -f "$MODEL" ] || { echo "[serve] no model at $MODEL (is its directory mounted at /models?)"; exit 2; }
case $MODEL in *-00001-of-00009.gguf) ;; *) say "warning: expected the first of 9 shards (*-00001-of-00009.gguf)";; esac
MISSING=0; for i in 2 3 4 5 6 7 8 9; do [ -f "${MODEL/00001-of/0000$i-of}" ] || { say "missing shard ${MODEL/00001-of/0000$i-of}"; MISSING=1; }; done
[ $MISSING = 0 ] || exit 2

# GPUs: the ones docker gave this container
FULL_MIB=97288   # what a whole RTX PRO 6000 reports free (97,887 MiB less the driver's own reserve); the presets are sized against it
mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,name,memory.total,memory.free,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits)
NGPU=${#GPU_LINES[@]}
[ "$NGPU" -ge 1 ] || { echo "[serve] no GPU visible (docker run --gpus ...)"; exit 2; }
[ "$NGPU" -le 2 ] || fail "$NGPU GPUs visible: the recipes are for one or two (pass --gpus '\"device=0,1\"')"
FREE=()
for l in "${GPU_LINES[@]}"; do
  IFS=, read -r idx name total free gen width <<< "$l"; name=$(echo "$name" | sed 's/^ *//')
  say "GPU $idx: $name, ${total// /} MiB, ${free// /} MiB free, PCIe Gen${gen// /} x${width// /}"
  case $name in *"RTX PRO 6000"*) ;; *) fail "GPU $idx is not an RTX PRO 6000 (the recipes are sized for 96 GB cards)";; esac
  [ "${gen// /}" -ge 5 ] 2>/dev/null && [ "${width// /}" -ge 16 ] 2>/dev/null || say "warning: GPU $idx links at Gen${gen// /} x${width// /}: prompt speed scales with the PCIe link (measured at Gen5 x16)"
  FREE+=("${free// /}")
done
if [ "$NGPU" = 2 ]; then
  P2P=$(nvidia-smi topo -p2p r 2>/dev/null | awk 'NR == 2 {print $3}')   # row GPU0, column GPU1
  [ "$P2P" = OK ] || fail "the two GPUs report no peer-to-peer read access (nvidia-smi topo -p2p r: '${P2P:-?}'); the two-card recipes move activations card to card"
fi

# the recipe: what every preset shares, what the card count adds, what the context changes
R=/app/recipes/${NGPU}gpu-${PRESET}.sh
[ -f "$R" ] || { echo "[serve] no recipe for $NGPU GPU(s) at --ctx $PRESET (have: $(ls /app/recipes | grep gpu- | sed 's/[.]sh$//' | tr '\n' ' '))"; exit 2; }
# shellcheck source=/dev/null
source /app/recipes/common.sh
# shellcheck source=/dev/null
source "/app/recipes/${NGPU}gpu.sh"
# shellcheck source=/dev/null
source "$R"

# the route bias: 0 unless asked for
if [ -n "$BIAS" ]; then
  awk -v v="$BIAS" 'BEGIN { exit !(v ~ /^[0-9]*[.]?[0-9]+$/) }' || { echo "[serve] --route-bias takes a number from 0 to 0.03, not '$BIAS'"; exit 2; }
  awk -v v="$BIAS" 'BEGIN { exit !(v + 0 <= 0.03) }' || fail "route bias $BIAS is above 0.03, the largest value measured"
  export LLAMA_MOE_ROUTE_BIAS=$BIAS
fi
if awk -v v="$LLAMA_MOE_ROUTE_BIAS" 'BEGIN { exit !(v + 0 > 0) }'; then
  say "route bias $LLAMA_MOE_ROUTE_BIAS: faster than 0, and the output is no longer the plain model's (README.md, \"The route bias\")"
fi

# host memory: the experts live in pinned host memory (~296 GiB) next to the engine's host buffers
MEM_AVAIL_GB=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
say "host memory: ${MEM_AVAIL_GB} GiB available; this preset peaked at ${NEED_RAM_GIB} GiB in use on the test host, the OS included"
[ "$MEM_AVAIL_GB" -ge "$NEED_RAM_GIB" ] || fail "not enough host memory: ${MEM_AVAIL_GB} GiB available, ${NEED_RAM_GIB} GiB needed (a host with 384 GiB of RAM or more)"
[ "$(ulimit -l)" = unlimited ] || fail "locked-memory limit is $(ulimit -l) (run the container with --ulimit memlock=-1:-1)"

# CPU threads: the physical cores
if [ -z "$THREADS" ]; then
  THREADS=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
  [ "$THREADS" -ge 1 ] 2>/dev/null || THREADS=$(nproc)
fi
NUMA=$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/ /, "", $2); print $2}')
[ "${NUMA:-1}" -le 1 ] || say "warning: $NUMA NUMA nodes: measured on one; pin the container to one node's cores and memory (--cpuset-cpus, --cpuset-mems) if speed is low"

# the MoE cache: the preset's measured size.  The presets fill a whole card to within about 0.5 GiB, so a card with less free (a desktop,
# another process) gives the difference up from the cache: all of it on one card; on two cards by the card's share of the cache
# (LLAMA_MOE_CACHE_SPLIT), or twice the larger difference where the engine splits the cache by free memory.  Rounded up to 1,000 MiB,
# plus 1,000 MiB (the cache moves in steps of about that size).
if [ -z "$CACHE" ]; then
  CUT=$(awk -v n="$NGPU" -v full="$FULL_MIB" -v f0="${FREE[0]}" -v f1="${FREE[1]:-0}" -v sp="${LLAMA_MOE_CACHE_SPLIT:-}" 'BEGIN {
    d0 = full - f0; if (d0 < 16) d0 = 0
    d1 = n == 2 ? full - f1 : 0; if (d1 < 16) d1 = 0
    if (n == 1) c = d0
    else if (split(sp, s, ":") == 2 && s[1] > 0 && s[2] > 0) { c0 = d0 * (s[1] + s[2]) / s[1]; c1 = d1 * (s[1] + s[2]) / s[2]; c = c0 > c1 ? c0 : c1 }
    else c = 2 * (d0 > d1 ? d0 : d1)
    if (c > 0) c = int((c + 999) / 1000) * 1000 + 1000
    printf "%d", c }')
  CACHE=$((CACHE_MIB - CUT))
  if [ "$CUT" -gt 0 ]; then
    [ $((CUT * 4)) -le "$CACHE_MIB" ] || fail "the GPU(s) have ${FREE[*]} MiB free, a whole card has $FULL_MIB: the MoE cache would go from $CACHE_MIB to $CACHE MiB, too far from what was measured (free the GPUs, or set --cache-mib N yourself)"
    [ "$CACHE" -ge 1000 ] || { echo "[serve] no room for a MoE cache (free the GPUs)"; exit 2; }
    say "warning: the GPU(s) have ${FREE[*]} MiB free, a whole card has $FULL_MIB: MoE cache $CACHE_MIB -> $CACHE MiB (a size that was not measured; expect less speed)"
  fi
else
  say "MoE cache set by hand: $CACHE MiB (the preset measured $CACHE_MIB)"
fi
say "recipe $(basename "$R" .sh), route bias $LLAMA_MOE_ROUTE_BIAS, MoE cache $CACHE MiB, $THREADS threads"

ARGS=(--model "$MODEL" --alias "$ALIAS" --host 0.0.0.0 --port 8080 --threads "$THREADS" --threads-batch "$THREADS"
      --moe-cache-mib "$CACHE" "${COMMON_ARGS[@]}" "${RECIPE_ARGS[@]}" "${PASS[@]}")
if [ $DRY = 1 ]; then
  env | grep -E '^(LLAMA_|GGML_|CUDA_SCALE|OMP_)' | sort
  echo /app/llama-server "${ARGS[@]}"
  exit 0
fi
exec /app/llama-server "${ARGS[@]}"
