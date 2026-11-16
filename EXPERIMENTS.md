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
- The hit rate per layer and the routing of the layers on the cards came with E2.

## E2. Which layers sit on the cards (2026-11-15, measured): a trade, not kept

The engine writes every host layer's routing with `LLAMA_MOE_CACHE_TRACE`. Two runs of the 24 decode prompts, one as shipped
and one with the opposite layers on the cards, gave the routing of all 71 layers that can move (8 and 75 to 77 cannot). A
replay of the cache on those traces (within 3 points of the measured hit rate) gave a hit rate per layer:

- **The layers differ little:** 50 to 67 %. Layers 3 to 14 and 24 to 26 are among the lowest (50 to 58 %); card 1's are all 60 to 63 %.
- So the pick is layers 3 to 14 and 24 to 26 on card 0, with card 1 as shipped.

| Run | Layers on the cards | Cache hit rate | Decode, tok/s |
|---|---|---|---|
| A | as shipped, trace on | 57.35 % | 23.5 |
| D | as shipped, plain | 57.35 % | 26.6 |
| B | the opposite set, trace on | 57.92 % | 26.4 |
| C | the pick, trace on | 60.38 % | 28.1 |
| C2 | the pick, plain, `scripts/bench.sh` in full | not logged | 27.6 |

| | As shipped | The pick |
|---|---|---|
| Decode, tok/s | 26.6 and 26.6 | 28.1 and 27.6 |
| Prompt to 3.6K / 100K / 201K, tok/s | 849 / 1,494 / 1,276 | 785 / 1,334 / 1,189 |
| First token after those steps, s | 4.2 / 64.9 / 78.7 | 4.6 / 72.6 / 84.5 |
| First token of a fresh 16K / 32K prompt, s | 11.3 / 21.1 | 12.7 / 23.8 |

- **The pick decodes 4 to 6 % faster and reads prompts 7 to 12 % slower.** The CPU computes 8.7 % fewer experts. The prompt
  cost was not traced; the pick leaves the host layers in long runs (15 to 23, 27 to 39), where the shipped set spreads them.
- **Not kept.** This host is used with long prompts, and the gain is near the spread between two runs of one preset (0.92 to
  1.05 a prompt). The preset is unchanged.
- **The choice of layers alone does little:** the opposite set scored the same as the shipped one (26.4 and 26.6).
- **Run A was slow by 12 %** with the same hit and miss counts as run D, to the expert. It was the first start after the
  memory-mapped one-card trial. The cause was not traced; the README's 26.6 was reproduced by run D.
- The shipped decode figures are the run of 2026-11-14 and run D; its prompt figures are the run of 2026-11-14 alone.

Raw files: `results/experiments/e2-*.json`. The replay is `bench/replay_cache.py`, and `bench/replay_hotmap.py` for E3. The
traces (27 MB each) are not in the repository: `LLAMA_MOE_CACHE_TRACE=<file>` on the server writes one.

## E3. What a hot-expert map would give (2026-11-15, simulated, not built)

The same replay with the DeepSeek layout: the hottest 87 experts of every layer on the cards only, the other 169 on the host
with the cache over them. That is the same GPU memory and the same host memory as today. The hot sets come from prompts 1 to
12 and the score from prompts 13 to 24.

| Layout | Expert traffic on the cards |
|---|---|
| Today: 24 whole layers of 71, and the cache | 73.5 to 73.9 % |
| Hot 87 a layer, and the cache | 79.8 to 80.1 % |
| Hot 110 a layer, a third of the cache | 77.5 to 77.8 % |
| Hot 130 a layer, almost no cache | 71.6 to 72.0 % |

- **About a quarter less CPU work** (26 % of the traffic to 20 %). The pick of E2 cut 8.7 % and gave 4 to 6 % decode, so this
  is perhaps 10 to 15 % decode. That is an estimate from a model that reads 3 points high, not a measurement.
- **GLM-5.3's routing is flatter than DeepSeek's.** The hottest 34 % of a layer's experts take half its traffic here; there
  57 % of the experts took 85 %. And this engine already has a cache that follows the conversation: it hits 57 % with 13.5 %
  of the host experts, where the same number of fixed hot experts would take about 30 %.
- **It needs an engine change:** the loader and the cache place whole tensors, and a cached expert is a copy of a pinned one.
  Its effect on prompts is unknown.

## What is left

1. **The hot-expert map of E3,** in the engine. The only candidate with a double-digit estimate, and the largest work.
2. **The route bias** (`--route-bias`), upstream's speed option, was not measured on this host. It changes the output.
3. **A spread pick:** the low layers on the cards but not in one block, to keep the prompt speed. Worth 2 to 3 % decode by the
   replay, which one run cannot resolve.
