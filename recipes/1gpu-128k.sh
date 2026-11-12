# 1gpu-128k.sh -- one RTX PRO 6000 (96 GB), 128K context.
# Prompt and decode as 1gpu-64k, with a smaller cache: more of it sits on memory the KV cache takes back as the context fills.
CTX=131072
CACHE_MIB=29800          # the MoE cache's own budget, as measured with every card whole
NEED_RAM_GIB=339         # host memory in use at this preset's peak on the test host, the OS included
RECIPE_ARGS=(-c "$CTX" -ub 16384 -b 65536 --spec-draft-n-ubatch 512)
export GGML_GALLOCR_HEADROOM_PCT=15
export LLAMA_MOE_PROMPT_PREFETCH=64
export LLAMA_UB_BALANCE=1
export LLAMA_MOE_CACHE_BORROW_MIB=37000      # cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_MOE_CACHE_LEND_MIB=7000         # cache slots on memory the KV cache takes back as the context fills
export LLAMA_SHORT_BATCH_UB=16
