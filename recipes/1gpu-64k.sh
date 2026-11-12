# 1gpu-64k.sh -- one RTX PRO 6000 (96 GB), 64K context.
# Prompt: ubatches of up to 16K tokens split evenly, the rarely-routed experts of each prompt ubatch computed on the CPU (predicted
# from the router's history, the next layer's experts fetched during the current layer's attention), every other expert streamed over
# PCIe.  Up to 128 new tokens stay on the decode path.
CTX=65536
CACHE_MIB=38500          # the MoE cache's own budget, as measured with every card whole
NEED_RAM_GIB=336         # host memory in use at this preset's peak on the test host, the OS included
RECIPE_ARGS=(-c "$CTX" -ub 16384 -b 65536 --spec-draft-n-ubatch 512)
export GGML_GALLOCR_HEADROOM_PCT=15
export LLAMA_MOE_PROMPT_PREFETCH=64
export LLAMA_UB_BALANCE=1
export LLAMA_MOE_CACHE_BORROW_MIB=31336      # cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_MOE_CACHE_LEND_MIB=3000         # cache slots on memory the KV cache takes back as the context fills
export LLAMA_SHORT_BATCH_UB=16
