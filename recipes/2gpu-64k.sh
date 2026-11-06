# 2gpu-64k.sh -- two RTX PRO 6000 (96 GB each, peer-to-peer), 64K context.
# Prompt: from 8K tokens the layer-major graph (each layer's experts go up once for 4K-token attention blocks, both cards pipelined);
# shorter prompts run both cards on every layer (the experts split by what each card caches).  Decode: layers split across the cards.
CTX=65536
FIXED_MIB=77600          # VRAM both cards need besides the MoE cache (models 17.3 GB, KV 3.2, compute 2 x ~14, staircases, CUDA, headroom)
RECIPE_ARGS=(-c "$CTX" -ub 16384 -b 131072)
export GGML_CUDA_P2P=1
export LLAMA_LAYER_MAJOR=4096
export LLAMA_LM_PIPE=6
export LLAMA_LM_MOE_ROWS=16384
export LLAMA_LM_MIN_UB=8192
export GGML_SCHED_COPIES=2
export LLAMA_TP_PROMPT=1
export LLAMA_TP_CUT=0.6
export LLAMA_TP_SHADOW_HOST=1
export LLAMA_TP_EP_CACHE=1
export LLAMA_TP_EP_SHIFT=-6
export LLAMA_LM_MIN_TOKENS=8192
export LLAMA_LM_FALLBACK_UB=8192
