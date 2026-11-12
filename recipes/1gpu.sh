# 1gpu.sh -- what one card adds to common.sh, at every context.
export LLAMA_PP_ONE_DEVICE=1
export LLAMA_MOE_PROMPT_CPU=1
export LLAMA_MOE_PROMPT_PRED=1
export LLAMA_MOE_PROMPT_CPU_SCALE=0.7
export LLAMA_MOE_PROMPT_STAGE_ARENA=1
export GGML_CUDA_LID_Q8_FUSED=1
export GGML_CUDA_LID_KPB=-1
