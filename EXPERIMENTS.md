# Experiments

Branch `experiment/deepseek-v41-recipe`. The question: what carries over from
[hfyeomans/deepseek-v41-mxfp4-recipe-rtx6000pro](https://github.com/hfyeomans/deepseek-v41-mxfp4-recipe-rtx6000pro), which serves
DeepSeek-V4.1-Flash on this same machine at about 60 tok/s, to GLM-5.3 here at 26.6.

A change that works goes into the README, "What this fork changes", with why, its cost and its trade-off. What fails stays in
this file and in RESULTS.md.

## What that recipe does differently

It is another engine (SGLang with kt-kernel) and another model, so no code carries over. One idea does: its hot-expert map.

- **It chooses the GPU experts by how often they are routed to**, per layer, from recorded traffic: 219 of 384 experts a layer
  on the cards, 165 on the CPU. That took decode from 42 to 58 tok/s. The CPU held 42 % of the experts and went from 42 % of
  the routed traffic to 15 %. It is an exact permutation: the output did not change.
- **An expert is on a card or on the host, not both**, so the host holds only what the cards do not.
- Not transferable: the Engram tables, the MXFP4 kernels, expert deferral (an approximation tied to kt-kernel), four slots.

Here the cards hold whole layers (28 of 75 at 256K), chosen evenly and not by traffic, plus a cache of the other layers' hot
experts. The cache's experts are copies: the host pins all of those layers.

## E1. Where the expert traffic goes now (2026-11-15, measured)

`2gpu-256k-ram256` as it ships, with `LLAMA_MOE_CACHE_LOG_EVERY=64`. Twelve prompts, 192 tokens each, 15 windows of 64 tokens.

| | |
|---|---|
| Cache hit rate on the 47 host layers | 55.3 % (51.6 to 57.3 % a window) |
| Cache size | 13.3 and 13.8 % of the host experts a card |
| Misses computed on the CPU | all of them |
| Decode with the log on | 23.4 tok/s median (21.4 to 27.0) |

- **The CPU computes about 28 % of the expert traffic:** 47 of 75 layers are on the host, and 44.7 % of their selections miss.
  The 28 layers on the cards are 37 % of it and the cache 35 %. This counts every layer the same and includes the draft tokens.
- **The routing is skewed:** 13.5 % of the host experts take 55 % of the host layers' traffic. That is what a hot map uses.
- The log counts the host layers only (read from the patch: a layer on the cards has no cache group). Its count agrees: 845
  selections a step is 47 layers × 8 experts × 2.25 tokens a step, with the MTP head's drafts.
- Not measured: the hit rate per layer, and the routing of the 28 layers on the cards.

## Candidates, cheapest first

1. **Choose the resident layers by traffic.** No engine change. Record the routing per layer (`LLAMA_MOE_ROUTE_HIST`), keep the
   layers with the flattest routing whole on the cards, and leave the skewed ones to the cache. The gain depends on how much
   the layers differ, which is not known yet.
2. **Hot experts on the cards, per expert.** The DeepSeek recipe's idea itself: the hottest experts of every layer on the cards
   and not on the host. It needs an engine change: the loader and the cache place whole tensors today. Larger gain, larger work.
3. **The route bias** (`--route-bias`), upstream's speed option, was not measured on this host. It changes the output.

**One card is not helped by any of these.** The limit there is that the card and the host together do not hold the model.
