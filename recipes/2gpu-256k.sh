# 2gpu-256k.sh -- two RTX PRO 6000 (96 GB each, peer-to-peer), 256K context.
# Decode: both cards work on every layer; each computes half the attention heads and the experts it caches.
# Prompt: 129 to 8,191 new tokens run both cards on every layer; from 8,192 the layer-major graph (each layer's experts go up once
# for 4K-token attention blocks, both cards pipelined).  Up to 128 new tokens stay on the decode path.  The sparse-attention indexer
# runs on both cards, with the fused q8_0 kernel.  The cache's shares on the two cards are fixed (LLAMA_MOE_CACHE_SPLIT), not taken
# from the free memory of each.
CTX=262144
CACHE_MIB=72600          # the MoE cache's own budget, as measured with every card whole
NEED_RAM_GIB=354         # host memory in use at this preset's peak on the test host, the OS included
RECIPE_ARGS=(-c "$CTX" -ub 16384 -b 131072 --spec-draft-n-ubatch 512)
export LLAMA_LAYER_MAJOR=4096
export LLAMA_TP_PROMPT=1
export LLAMA_TP_CUT=0.6
export LLAMA_TP_SHADOW_HOST=1
export LLAMA_TP_EP_CACHE=1
export LLAMA_TP_EP_SHIFT=-6
export LLAMA_LM_MIN_TOKENS=8192
export LLAMA_LM_FALLBACK_UB=8192
export LLAMA_MOE_CACHE_BORROW_MIB=27800      # cache slots on the compute buffer's memory, given up while a long prompt runs
export LLAMA_MOE_CACHE_BORROW_HEAD_MIB=512   # the compute buffer's private part (2,048 by default)
export GGML_CUDA_LID_Q8_FUSED=1
export GGML_CUDA_LID_KPB=-1
export LLAMA_TD_LID_REP=1
export LLAMA_TP_PROMPT_MIN=129
export LLAMA_MOE_CACHE_SPLIT=0.5281:0.4719   # the cache's shares on card 0 : card 1
export LLAMA_TP_DECODE=1
export LLAMA_HEAD_SPLIT=1
export GGML_CUDA_PAIR_AR_P2P=1
export LLAMA_MOE_CACHE_LEND_MIB=13000        # cache slots on memory the KV cache takes back as the context fills
export LLAMA_TD_AR_ADD=1
export LLAMA_HEAD_OVERLAP=1
