# common.sh -- settings every preset shares.  serve.sh sources this file, then <N>gpu.sh (what the card count adds), then
# <N>gpu-<ctx>.sh (what the context changes).  Each setting is a measured win or a requirement of one; see RESULTS.md.
#
# The model's 78 MoE layers keep their routed experts in pinned host memory (-ot ... =CPU); the GPU holds everything else plus a
# cache of experts (--moe-cache-mib).  The NextN / MTP layer (78) stays whole on the GPU: it drafts 2 tokens a step (draft-mtp).

# routed experts of layers 0-77 in host memory; layer 78 (the MTP head) on the GPU
OT='blk[.]([0-9]|[1-6][0-9]|7[0-7])[.]ffn_(up|down|gate|gate_up)_(ch|)exps=CPU'

COMMON_ARGS=(
  -ngl 999 -sm layer -fa on --jinja --parallel 1 --metrics --no-warmup
  -ot "$OT"
  --moe-cache-cpu-share 0.9          # once the cache is full, 90 % of the misses are computed on the CPU instead of uploaded
  --load-mode none
  -ctk q8_0 -ctv q8_0
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling
)

# The route bias.  0 (the default): the router picks the experts exactly as the model defines.  A value above 0 adds that much to the
# router score of the experts the cache already holds, so more tokens find their experts on the GPU.  That is faster, and it changes
# the output: an answer is no longer the plain model's, and it can depend on what the server handled before.  serve.sh --route-bias V
# sets it; 0.03 is the measured speed option (README.md, "The route bias").
export LLAMA_MOE_ROUTE_BIAS=0

# engine settings every preset runs with
export LLAMA_TWIN_BACKENDS=0
export CUDA_SCALE_LAUNCH_QUEUES=4x
export LLAMA_MOE_ASYNC_FILL=1
export LLAMA_MOE_SHEXP_OVERLAP=1
export LLAMA_MTP_MOE_CACHE_MIB=0
export GGML_SCHED_GALLOCS=1
export GGML_CUDA_LID_MMA=2
export LLAMA_MTP_DSA=1
export LLAMA_MOE_WEIGHT_INPLACE=1
export LLAMA_MOE_CACHE_LFU=64
export LLAMA_MOE_UPLOAD_RING=1
export LLAMA_MOE_UPLOAD_RING_MEMCPY=1
export GGML_CUDA_RING_MIXED=1
export GGML_SCHED_ASYNC_HOST_INPUT=2
export GGML_CUDA_Q8_1_DEDUPE=1
export LLAMA_MOE_ASYNC_FILL_TOKENS=8
export LLAMA_MOE_DEFERRED_PLAN=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export GGML_SCHED_COPIES=2
export GGML_SCHED_PAIR_GATE_UP=4097
export GGML_CUDA_FUSE_IQ3_XXS=1
export GGML_CUDA_FA_MLA_SPARSE=1
export LLAMA_MOE_ZERO_SKIP=1
export GGML_CUDA_MMVQ_SMALLK_MT=2
export LLAMA_DSA_SHORT_DENSE=1
export GGML_CUDA_FA_GATHER_K=1
export GGML_CUDA_SPARSE_CHUNKED=1
export GGML_CUDA_LID_SKIP_IDLE=1
export LLAMA_MOE_CACHE_BORROW_RING=1
export LLAMA_KV_LAZY=1
export LLAMA_GRAPH_DUAL_SCHED=1
export LLAMA_KV_SEQ_RM_BOUNDED=1
export LLAMA_MTP_HEAD_TYPE=q4_K
export LLAMA_SYNC_SKIP_IDLE=1
export LLAMA_MTP_SHORT_DENSE=1
export LLAMA_MOE_CACHE_EQUAL_COVER=1
export LLAMA_SHORT_BATCH_MAX=128
