# 2gpu-1m-ram256.sh -- two RTX PRO 6000 (96 GB each), a 256 GB host, 1M context (1,000,000 tokens, as 1gpu-1m).
# The KV cache takes 25 GiB a card, so 20 layers' experts fit on the cards (11 on card 0, 9 on card 1); the other 55 layers' are in
# pinned host memory (214.0 GiB).  That left the test host 9.7 GiB at the lowest: nothing else of size can run on it.  Memory-mapping
# the model instead (LOAD_MODE=mmap) is safe on memory and too slow on prompts: see RESULTS.md, "A 256 GB host".
# Prompt: 512-token attention blocks and ubatches of 2K tokens, in batches of 16K.  4K ubatches (about 1 GiB more compute buffer a
# card) were not tried at this context.
CTX=1000000
CACHE_MIB=4600           # the MoE cache's own budget; the rest of the cache sits on the compute buffer's and the KV cache's memory
NEED_RAM_GIB=238         # host memory in use at this preset's peak on the 256 GB test host, the OS included
CACHE_RAM_MIB=2048      # the prompt cache: up to 41K tokens in all; the host has no room for more at this context
RESIDENT="3 6 8 9 12 15 18 21 24 27 30  50 54 58 62 66 70 75 76 77"
RECIPE_ARGS=(-c "$CTX" -ub 2048 -b 16384 -ts 1,1 --spec-draft-n-ubatch 128)
export LLAMA_LAYER_MAJOR=512
export LLAMA_LM_MIN_UB=2048
export LLAMA_LM_MOE_ROWS=2048
export LLAMA_MOE_CACHE_BORROW_MIB=11500      # a card: cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_MOE_CACHE_LEND_MIB=26000        # a card: cache slots on memory the KV cache takes back as the context fills
export LLAMA_MOE_CACHE_SPLIT=0.39:0.61       # the cache's own budget on card 0 : card 1
