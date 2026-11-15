# 2gpu-256k-ram256.sh -- two RTX PRO 6000 (96 GB each), a 256 GB host, 256K context.
# 28 layers' experts on the cards (15 on card 0, 13 on card 1), 47 layers' in pinned host memory (182.9 GiB).
# Prompt: 1K-token attention blocks and ubatches of 4K tokens, as 2gpu-512k-ram256: on the test host that was as fast as 2K blocks
# and 16K ubatches, with 10 GiB less compute buffer a card and half the host buffers.
CTX=262144
CACHE_MIB=11000          # the MoE cache's own budget; the rest of the cache sits on the compute buffer's and the KV cache's memory
NEED_RAM_GIB=219         # host memory in use at the peak, the OS included: 211 GiB measured with an 8 GiB prompt cache, and 8 more for this one
                         # (a run with this one peaked at 216.3 GiB with 14.5 GiB in it)
CACHE_RAM_MIB=16384     # the prompt cache: any one conversation the context allows (12.7 GiB at 262,144 tokens) survives a switch to
                         # another one, and up to 330K tokens in all; 8192 stops at 165K and gives 8 GiB of host memory back
RESIDENT="3 5 6 8 9 12 15 18 21 24 27 30 33 36 39  42 44 46 50 54 58 62 66 70 72 75 76 77"
RECIPE_ARGS=(-c "$CTX" -ub 4096 -b 32768 -ts 1,1 --spec-draft-n-ubatch 128)
export LLAMA_LAYER_MAJOR=1024
export LLAMA_LM_MIN_UB=4096
export LLAMA_LM_MOE_ROWS=4096
export LLAMA_MOE_CACHE_BORROW_MIB=13500      # a card: cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_MOE_CACHE_LEND_MIB=7000         # a card: cache slots on memory the KV cache takes back as the context fills
export LLAMA_MOE_CACHE_SPLIT=0.455:0.545     # the cache's own budget on card 0 : card 1
