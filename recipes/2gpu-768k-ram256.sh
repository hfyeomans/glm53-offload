# 2gpu-768k-ram256.sh -- two RTX PRO 6000 (96 GB each), a 256 GB host, 768K context (786,432 tokens).
# The KV cache takes 20 GiB a card, so 22 layers' experts fit on the cards (12 on card 0, 10 on card 1); the other 53 layers' are
# in pinned host memory (206.2 GiB).
# Prompt: 512-token attention blocks and ubatches of 4K tokens.
CTX=786432
CACHE_MIB=8500           # the MoE cache's own budget; the rest of the cache sits on the compute buffer's and the KV cache's memory
NEED_RAM_GIB=233         # host memory in use at the peak, the OS included: 231 GiB measured with a 2 GiB prompt cache, and 2 more for this one
CACHE_RAM_MIB=4096      # the prompt cache: up to 83K tokens in all; the host has no room for more at this context
RESIDENT="3 6 8 9 12 15 18 21 24 27 30 33  46 50 54 58 62 66 70 75 76 77"
RECIPE_ARGS=(-c "$CTX" -ub 4096 -b 32768 -ts 1,1 --spec-draft-n-ubatch 128)
export LLAMA_LAYER_MAJOR=512
export LLAMA_LM_MIN_UB=4096
export LLAMA_LM_MOE_ROWS=4096
export LLAMA_MOE_CACHE_BORROW_MIB=11500      # a card: cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_MOE_CACHE_LEND_MIB=20000        # a card: cache slots on memory the KV cache takes back as the context fills
export LLAMA_MOE_CACHE_SPLIT=0.41:0.59       # the cache's own budget on card 0 : card 1
