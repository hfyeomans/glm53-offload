# 2gpu-1m-ram256-hot.sh -- what the hot-expert map changes in 2gpu-1m-ram256.sh (EXPERIMENTS.md, E6); on by default, --no-hot-map turns it off.
# The preset keeps 20 layers' experts whole on the cards.  With the map only layers 8 and 75-77 stay whole; the memory of the other
# 16 holds the busiest experts of every layer of the card instead (hotmap-1m.txt: 2,560 on card 0, 1,536 on card 1, pinned in the
# expert cache and never in host memory).  Host memory holds the other 14,080 experts, as the preset.
RESIDENT="8 75 76 77"
CACHE_PIN_MIB=63744      # the pinned experts: 4,096 of 15.5625 MiB
CACHE_MIB=68344         # the preset's 4,600 MiB and the pinned experts
export LLAMA_MOE_CACHE_SPLIT=0.6092:0.3908    # card 0 : card 1, each card's 1,794 / 2,806 MiB of the preset and its pinned experts
export LLAMA_MOE_HOT_MAP=/app/recipes/hotmap-1m.txt
