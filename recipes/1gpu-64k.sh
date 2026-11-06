# 1gpu-64k.sh -- one RTX PRO 6000 (96 GB), 64K context.
# Prompt: ubatches of 16K tokens split evenly, the rarely-routed experts of each prompt ubatch computed on the CPU (predicted from the
# router's history, the next layer's experts fetched during the current layer's attention), every other expert streamed over PCIe.
CTX=65536
FIXED_MIB=56000          # VRAM this shape needs besides the MoE cache (model 17.3 GB, KV 3.2, compute 17.3, MTP 0.6, CUDA, 2.4 GB headroom)
RECIPE_ARGS=(-c "$CTX" -ub 16384 -b 65536)
export LLAMA_PP_ONE_DEVICE=1
export LLAMA_MOE_PROMPT_CPU=1
export LLAMA_MOE_PROMPT_PRED=1
export LLAMA_MOE_PROMPT_CPU_SCALE=0.7
export LLAMA_MOE_PROMPT_PREFETCH=64
export GGML_SCHED_COPIES=2
export LLAMA_UB_BALANCE=1
