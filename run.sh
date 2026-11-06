#!/bin/bash
# run.sh -- serve GLM-5.3 on one or two RTX PRO 6000 from this repo's image.
#
#   ./run.sh --model /path/to/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf [--gpus 0 | 0,1] [--port 8080] [--bind 127.0.0.1]
#            [--image glm53-offload:latest] [serve.sh options, e.g. --cache-mib N --threads N --dry-run] [-- llama-server args]
#
# Default GPUs: 0 (one card); --gpus 0,1 for two; --detach runs it in the background (container --name glm53-offload).
# The API comes up at http://<bind>:<port>/v1 once the model has loaded (1-3 min
# from a warm page cache, longer from disk).  Ctrl-C stops it.
set -u
MODEL=; GPUS=0; PORT=8080; BIND=127.0.0.1; IMAGE=${GLM53_IMAGE:-glm53-offload:latest}; NAME=glm53-offload; DETACH=0; REST=()
while [ $# -gt 0 ]; do case $1 in
  --model) MODEL=$2; shift 2;;
  --gpus) GPUS=$2; shift 2;;
  --port) PORT=$2; shift 2;;
  --bind) BIND=$2; shift 2;;
  --image) IMAGE=$2; shift 2;;
  --name) NAME=$2; shift 2;;
  --detach) DETACH=1; shift;;
  -h|--help) sed -n 2,9p "$0"; exit 0;;
  *) REST+=("$1"); shift;;
esac; done
[ -n "$MODEL" ] || { echo "run.sh: --model is required (see --help)"; exit 2; }
[ -f "$MODEL" ] || { echo "run.sh: no file at $MODEL"; exit 2; }
command -v docker >/dev/null || { echo "run.sh: docker is not installed"; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "run.sh: no image $IMAGE (docker build -t glm53-offload -f docker/Dockerfile .)"; exit 2; }
DIR=$(cd "$(dirname "$MODEL")" && pwd); BASE=$(basename "$MODEL")
MODE=(--rm); if [ $DETACH = 1 ]; then MODE=(-d); elif [ -t 0 ]; then MODE=(--rm -it); fi
# --init: tini as PID 1; without it a killed llama-server was left a zombie and `docker rm -f` failed (2026-11-06)
docker run "${MODE[@]}" --init --name "$NAME" --gpus "\"device=$GPUS\"" --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -p "$BIND:$PORT:8080" -v "$DIR":/models:ro "$IMAGE" --model "/models/$BASE" "${REST[@]}"
rc=$?
[ $DETACH = 1 ] && [ $rc = 0 ] && echo "run.sh: started container $NAME (docker logs -f $NAME; docker rm -f $NAME to stop); API at http://$BIND:$PORT/v1 once loaded"
exit $rc
