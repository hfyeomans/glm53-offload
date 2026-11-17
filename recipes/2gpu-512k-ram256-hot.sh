# 2gpu-512k-ram256-hot.sh -- what the hot-expert map changes in 2gpu-512k-ram256.sh (EXPERIMENTS.md, E6); on by default, --no-hot-map turns it off.
# The preset keeps 24 layers' experts whole on the cards.  With the map only layers 8 and 75-77 stay whole; the memory of the other
# 20 holds the busiest experts of every layer of the card instead (hotmap-512k.txt: 3,072 on card 0, 2,048 on card 1, pinned in the
# expert cache and never in host memory).  Host memory holds the other 13,056 experts, as the preset.
RESIDENT="8 75 76 77"
CACHE_PIN_MIB=79680      # the pinned experts: 5,120 of 15.5625 MiB
CACHE_MIB=87880         # the preset's 8,200 MiB and the pinned experts
export LLAMA_MOE_CACHE_SPLIT=0.5804:0.4196    # card 0 : card 1, each card's 3,198 / 5,002 MiB of the preset and its pinned experts
export LLAMA_MOE_HOT_MAP=/app/recipes/hotmap-512k.txt
