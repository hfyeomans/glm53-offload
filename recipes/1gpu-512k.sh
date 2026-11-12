# 1gpu-512k.sh -- one RTX PRO 6000 (96 GB), 512K context.
# Prompt: from 513 new tokens the layer-major graph, as 1gpu-256k.
CTX=524288
CACHE_MIB=14400          # the MoE cache's own budget, as measured with every card whole
NEED_RAM_GIB=342         # host memory in use at this preset's peak on the test host, the OS included
RECIPE_ARGS=(-c "$CTX" -ub 16384 -b 131072 --spec-draft-n-ubatch 512)
export GGML_GALLOCR_HEADROOM_PCT=15
export LLAMA_MOE_PROMPT_PREFETCH=64
export LLAMA_LAYER_MAJOR=2048
export LLAMA_LM_PIPE=6
export LLAMA_LM_MIN_UB=8192
export LLAMA_LM_MOE_ROWS=16384
export LLAMA_MOE_CACHE_LEND_MIB=26900        # cache slots on memory the KV cache takes back as the context fills
export LLAMA_MOE_CACHE_BORROW_MIB=29000      # cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_SHORT_BATCH_UB=16
export LLAMA_LM_MIN_TOKENS=513
