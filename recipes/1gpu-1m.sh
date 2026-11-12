# 1gpu-1m.sh -- one RTX PRO 6000 (96 GB), 1M context (1,000,000 tokens).
# Prompt: from 513 new tokens the layer-major graph, in 512-token attention blocks and ubatches of 4K tokens.
# The KV cache and the compute buffer take most of the card at this context, so the MoE cache is small: the card must be whole.
CTX=1000000
CACHE_MIB=5400           # the MoE cache's own budget, as measured with every card whole
NEED_RAM_GIB=338         # host memory in use at this preset's peak on the test host, the OS included
RECIPE_ARGS=(-c "$CTX" -ub 4096 -b 131072 --spec-draft-n-ubatch 128)
export GGML_GALLOCR_HEADROOM_PCT=5
export LLAMA_MOE_PROMPT_PREFETCH=32
export LLAMA_LAYER_MAJOR=512
export LLAMA_LM_PIPE=6
export LLAMA_LM_MIN_UB=4096
export LLAMA_LM_MOE_ROWS=4096
export LLAMA_MOE_CACHE_LEND_MIB=50000        # cache slots on memory the KV cache takes back as the context fills
export LLAMA_MOE_CACHE_BORROW_MIB=16000      # cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_SHORT_BATCH_UB=4
export LLAMA_LM_MIN_TOKENS=513
