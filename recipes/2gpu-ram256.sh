# 2gpu-ram256.sh -- what a 256 GB host adds to common.sh and 2gpu.sh, at every context (serve.sh --ram 256).
#
# The 384 presets pin the routed experts of all 75 MoE layers in host memory (296 GiB).  A 256 GB host has about 247 GiB, so these
# presets keep the experts of some layers whole on their card (RESIDENT, in each preset) and pin the rest: 183 to 214 GiB.  Those
# layers never miss.  The GPU memory they take comes out of the expert cache, which now serves only the other layers.
#
# Decode: the layers split across the cards, as 2gpu-1m (the both-cards decode keeps a second copy of the KV cache and was not
# tried with resident layers).  Prompt: from 513 new tokens the layer-major graph, both cards pipelined.
#
# A card's memory, per preset: the weights (6.2 GiB on card 0, 10.7 on card 1), the resident layers (3,984 MiB each), the KV cache
# (6.4 and 6.0 GiB per 262,144 tokens), the layer-major staircase, the compute buffer and the upload ring (two slots of the card's
# largest layer: 10.5 and 9.5 GiB), about 2 GiB of driver and prompt-time memory, and the cache's own slots in what is left.
# Layers 8 and 75-77 are larger quants and are always resident: on the host they would be slot groups of their own in the cache,
# and a group needs a few GiB of the cache's own budget (with less, the engine does not cache the card's layers and stops).
# Layers 0-39 are on card 0 and 40-78 on card 1 (-ts 1,1 in each preset), so a RESIDENT list says how much each card holds.
# CACHE_RAM_MIB, in each preset: the server's prompt cache (--cache-ram).  When a request does not continue the conversation in the
# slot, the server saves that conversation's state to host memory, about 1 GiB per 20,600 tokens, and restores it when a later
# request returns to it.  A state larger than the cache is not saved and is computed again.  The cache is host memory next to the
# pinned experts, so each context gets what the host has left; NEED_RAM_GIB counts it full.  serve.sh --prompt-cache-mib N sets it.
CACHE_RAM_MEASURED_MIB=2048   # every preset's host memory was measured with a prompt cache of this size: with a smaller
                              # --prompt-cache-mib the memory check falls with the cache, and stops at this size
FULL_MIB=97100           # measured with a 150 MiB desktop on card 0: a card with this much free counts as whole
export GGML_CUDA_LID_Q8_FUSED=1
export GGML_CUDA_LID_KPB=-1
export LLAMA_LM_MIN_TOKENS=513
