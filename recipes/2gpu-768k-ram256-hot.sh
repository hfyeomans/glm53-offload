# 2gpu-768k-ram256-hot.sh -- what the hot-expert map changes in 2gpu-768k-ram256.sh (EXPERIMENTS.md, E6); on by default, --no-hot-map turns it off.
# The preset keeps 22 layers' experts whole on the cards.  With the map only layers 8 and 75-77 stay whole; the memory of the other
# 18 holds the busiest experts of every layer of the card instead (hotmap-768k.txt: 2,816 on card 0, 1,792 on card 1, pinned in the
# expert cache and never in host memory).  Host memory holds the other 13,568 experts, as the preset.
RESIDENT="8 75 76 77"
CACHE_PIN_MIB=71712      # the pinned experts: 4,608 of 15.5625 MiB
CACHE_MIB=80212         # the preset's 8,500 MiB and the pinned experts
export LLAMA_MOE_CACHE_SPLIT=0.5898:0.4102    # card 0 : card 1, each card's 3,485 / 5,015 MiB of the preset and its pinned experts
export LLAMA_MOE_HOT_MAP=/app/recipes/hotmap-768k.txt
