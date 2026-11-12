# 2gpu-1m.sh -- two RTX PRO 6000 (96 GB each, peer-to-peer), 1M context (1,048,576 tokens).
# Decode: the layers split across the cards (the both-cards decode is not used at this context).
# Prompt: from 513 new tokens the layer-major graph (2K-token attention blocks, both cards pipelined).
CTX=1048576
CACHE_MIB=59050          # the MoE cache's own budget, as measured with every card whole
NEED_RAM_GIB=342         # host memory in use at this preset's peak on the test host, the OS included
RECIPE_ARGS=(-c "$CTX" -ub 16384 -b 131072 --spec-draft-n-ubatch 128)
export LLAMA_LAYER_MAJOR=2048
export LLAMA_MOE_CACHE_BORROW_MIB=22700      # cache slots on the compute buffer's memory, given up while a long prompt runs
export GGML_CUDA_LID_Q8_FUSED=1
export GGML_CUDA_LID_KPB=-1
export LLAMA_LM_MIN_TOKENS=513
export LLAMA_MOE_CACHE_SPLIT=0.5415:0.4585   # the cache's shares on card 0 : card 1
export LLAMA_MOE_CACHE_LEND_MIB=26000        # cache slots on memory the KV cache takes back as the context fills
