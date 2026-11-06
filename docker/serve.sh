#!/bin/bash
# serve.sh -- the image's entrypoint: checks the machine, picks the recipe for the GPUs it was given (one or two RTX PRO 6000), sizes the
# MoE cache from the free VRAM and starts llama-server on port 8080 (OpenAI-compatible: /v1/chat/completions).
#
#   serve.sh --model /models/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf [options] [-- extra llama-server args]
#     --ctx 64k            context preset (64k is the measured one)
#     --cache-mib N        MoE cache size in MiB over all GPUs (default: free VRAM minus what the recipe needs besides the cache)
#     --threads N          CPU threads (default: the physical cores)
#     --alias NAME         model name the API reports (default glm-5.3)
#     --force              start even if a check fails (RAM, GPU model, peer-to-peer)
#     --dry-run            print the command and the environment, start nothing
set -u
MODEL=; PRESET=64k; CACHE=; THREADS=; ALIAS=glm-5.3; FORCE=0; DRY=0; PASS=()
while [ $# -gt 0 ]; do case $1 in
  --model) MODEL=$2; shift 2;;
  --ctx) PRESET=$2; shift 2;;
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
mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,name,memory.total,memory.free,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits)
NGPU=${#GPU_LINES[@]}
[ "$NGPU" -ge 1 ] || { echo "[serve] no GPU visible (docker run --gpus ...)"; exit 2; }
[ "$NGPU" -le 2 ] || fail "$NGPU GPUs visible: the recipes are for one or two (pass --gpus '\"device=0,1\"')"
FREE_SUM=0
for l in "${GPU_LINES[@]}"; do
  IFS=, read -r idx name total free gen width <<< "$l"; name=$(echo "$name" | sed 's/^ *//')
  say "GPU $idx: $name, ${total// /} MiB, ${free// /} MiB free, PCIe Gen${gen// /} x${width// /}"
  case $name in *"RTX PRO 6000"*) ;; *) fail "GPU $idx is not an RTX PRO 6000 (the recipes are sized for 96 GB cards)";; esac
  [ "${gen// /}" -ge 5 ] 2>/dev/null && [ "${width// /}" -ge 16 ] 2>/dev/null || say "warning: GPU $idx links at Gen${gen// /} x${width// /}: prompt speed scales with the PCIe link (measured at Gen5 x16)"
  FREE_SUM=$((FREE_SUM + ${free// /}))
done
if [ "$NGPU" = 2 ]; then
  P2P=$(nvidia-smi topo -p2p r 2>/dev/null | awk 'NR == 2 {print $3}')   # row GPU0, column GPU1
  [ "$P2P" = OK ] || fail "the two GPUs report no peer-to-peer read access (nvidia-smi topo -p2p r: '${P2P:-?}'); the two-card recipe moves activations card to card"
fi

# host memory: the experts live in pinned host memory (~304 GB) next to ~7-19 GB of host buffers
MEM_AVAIL_GB=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
NEED_GB=$([ "$NGPU" = 2 ] && echo 336 || echo 325)
say "host memory: ${MEM_AVAIL_GB} GiB available, the recipe pins ~${NEED_GB} GiB"
[ "$MEM_AVAIL_GB" -ge "$NEED_GB" ] || fail "not enough host memory: ${MEM_AVAIL_GB} GiB available, ~${NEED_GB} GiB needed (512 GB of RAM recommended)"
[ "$(ulimit -l)" = unlimited ] || fail "locked-memory limit is $(ulimit -l) (run the container with --ulimit memlock=-1:-1)"

# CPU threads: the physical cores
if [ -z "$THREADS" ]; then
  THREADS=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
  [ "$THREADS" -ge 1 ] 2>/dev/null || THREADS=$(nproc)
fi
NUMA=$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/ /, "", $2); print $2}')
[ "${NUMA:-1}" -le 1 ] || say "warning: $NUMA NUMA nodes: measured on one; pin the container to one node's cores and memory (--cpuset-cpus, --cpuset-mems) if speed is low"

# the recipe
R=/app/recipes/${NGPU}gpu-${PRESET}.sh
[ -f "$R" ] || { echo "[serve] no recipe for $NGPU GPU(s) at --ctx $PRESET (have: $(ls /app/recipes | grep gpu | tr '\n' ' '))"; exit 2; }
# shellcheck source=/dev/null
source /app/recipes/common.sh
# shellcheck source=/dev/null
source "$R"
if [ -z "$CACHE" ]; then
  CACHE=$(( (FREE_SUM - FIXED_MIB) / 500 * 500 ))
  [ "$CACHE" -ge 8000 ] || fail "only $FREE_SUM MiB free on the GPU(s): the MoE cache would be $CACHE MiB (free the GPUs, or --cache-mib N)"
fi
say "recipe $(basename "$R"), MoE cache $CACHE MiB, $THREADS threads"

ARGS=(--model "$MODEL" --alias "$ALIAS" --host 0.0.0.0 --port 8080 --threads "$THREADS" --threads-batch "$THREADS"
      --moe-cache-mib "$CACHE" "${COMMON_ARGS[@]}" "${RECIPE_ARGS[@]}" "${PASS[@]}")
if [ $DRY = 1 ]; then
  env | grep -E '^(LLAMA_|GGML_|CUDA_SCALE|OMP_)' | sort
  echo /app/llama-server "${ARGS[@]}"
  exit 0
fi
exec /app/llama-server "${ARGS[@]}"
