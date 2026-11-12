# 2gpu.sh -- what two cards add to common.sh, at every context.
export GGML_GALLOCR_HEADROOM_PCT=15
export GGML_CUDA_P2P=1
export LLAMA_LM_PIPE=6
export LLAMA_LM_MOE_ROWS=16384
export LLAMA_LM_MIN_UB=8192
export LLAMA_MOE_XD=1
export LLAMA_MOE_XD_REDUCE=1
export LLAMA_SHORT_BATCH_UB=4
