# 2gpu-512k-ram256.sh -- two RTX PRO 6000 (96 GB each), a 256 GB host, 512K context.
# 24 layers' experts on the cards (13 on card 0, 11 on card 1), 51 layers' in pinned host memory (198.4 GiB).
# Prompt: 1K-token attention blocks and ubatches of 4K tokens, so the compute buffer leaves the KV cache its 13 GiB a card.
CTX=524288
CACHE_MIB=8200           # the MoE cache's own budget; the rest of the cache sits on the compute buffer's and the KV cache's memory
NEED_RAM_GIB=227         # host memory in use at the peak, the OS included: 221 GiB measured with a 2 GiB prompt cache, and 6 more for this one
CACHE_RAM_MIB=8192      # the prompt cache: conversations of up to 165K tokens in all survive a switch to another one
RESIDENT="3 6 8 9 12 15 18 21 24 27 30 33 36  42 46 50 54 58 62 66 70 75 76 77"
RECIPE_ARGS=(-c "$CTX" -ub 4096 -b 32768 -ts 1,1 --spec-draft-n-ubatch 128)
export LLAMA_LAYER_MAJOR=1024
export LLAMA_LM_MIN_UB=4096
export LLAMA_LM_MOE_ROWS=4096
export LLAMA_MOE_CACHE_BORROW_MIB=13500      # a card: cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_MOE_CACHE_LEND_MIB=14000        # a card: cache slots on memory the KV cache takes back as the context fills
export LLAMA_MOE_CACHE_SPLIT=0.39:0.61       # the cache's own budget on card 0 : card 1
