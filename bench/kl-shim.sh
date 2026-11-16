#!/bin/bash
# mounted over /app/llama-server: serve.sh execs it with the preset's exact environment and arguments; it runs llama-perplexity
# instead, without the server-only arguments, with a 4,096-token context and 4-token batches (the batches the expert cache and the
# route bias serve, as in decode), plus $KL_ARGS
A=()
while [ $# -gt 0 ]; do case $1 in
  --host|--port|--alias|--cache-ram|--parallel|--spec-type|--spec-draft-n-max|--spec-draft-n-ubatch|-c|-ub|-b|-lv) shift 2;;
  --metrics|--jinja|--spec-draft-backend-sampling) shift;;
  *) A+=("$1"); shift;;
esac; done
echo "[shim] env:"; env | grep -E '^(LLAMA_MOE_ROUTE_BIAS|LLAMA_MOE_HOT_MAP|LLAMA_MOE_CACHE_SPLIT)=' | sed 's/^/[shim]   /'
echo "[shim] /app/llama-perplexity ${A[*]} -c 4096 -b 4 -ub 4 -lv 3 $KL_ARGS"
# shellcheck disable=SC2086
exec /app/llama-perplexity "${A[@]}" -c 4096 -b 4 -ub 4 -lv 3 $KL_ARGS
