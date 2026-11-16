# 2gpu-256k-ram256-hot.sh -- what the hot-expert map changes in 2gpu-256k-ram256.sh (EXPERIMENTS.md, E4); on by default, --no-hot-map turns it off.
# The preset keeps 28 layers' experts whole on the cards.  With the map only layers 8 and 75-77 stay whole; the memory of the other
# 24 holds the busiest experts of every layer of the card instead (hotmap-256k.txt: 3,584 on card 0, 2,560 on card 1, pinned in the
# expert cache and never in host memory).  Host memory holds the other 12,032 experts: 182.9 GiB, as the preset.
RESIDENT="8 75 76 77"
CACHE_PIN_MIB=95616      # the pinned experts: 6,144 of 15.5625 MiB
CACHE_MIB=106616         # the preset's 11,000 MiB and the pinned experts
export LLAMA_MOE_CACHE_SPLIT=0.5701:0.4299    # card 0 : card 1, each card's 5,005 / 5,995 MiB of the preset and its pinned experts
export LLAMA_MOE_HOT_MAP=/app/recipes/hotmap-256k.txt
