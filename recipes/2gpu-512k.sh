# 2gpu-512k.sh -- two RTX PRO 6000 (96 GB each, peer-to-peer), 512K context.
# Decode: both cards work on every layer, as 2gpu-64k.
# Prompt: from 513 new tokens the layer-major graph (4K-token attention blocks, both cards pipelined).  The both-cards prompt graph
# is not used at this context.  Up to 128 new tokens stay on the decode path.
CTX=524288
CACHE_MIB=50860          # the MoE cache's own budget, as measured with every card whole
NEED_RAM_GIB=343         # host memory in use at this preset's peak on the test host, the OS included
RECIPE_ARGS=(-c "$CTX" -ub 16384 -b 131072 --spec-draft-n-ubatch 512)
export LLAMA_LAYER_MAJOR=4096
export LLAMA_MOE_CACHE_BORROW_MIB=23300      # cache slots on the compute buffer's memory, given up while a long prompt runs
export GGML_CUDA_LID_Q8_FUSED=1
export GGML_CUDA_LID_KPB=-1
export LLAMA_TD_LID_REP=1
export LLAMA_LM_MIN_TOKENS=513
export LLAMA_MOE_CACHE_SPLIT=0.5606:0.4394   # the cache's shares on card 0 : card 1
export LLAMA_TP_DECODE=1
export LLAMA_HEAD_SPLIT=1
export GGML_CUDA_PAIR_AR_P2P=1
export LLAMA_MOE_CACHE_LEND_MIB=26000        # cache slots on memory the KV cache takes back as the context fills
export LLAMA_TD_AR_ADD=1
export LLAMA_HEAD_OVERLAP=1
