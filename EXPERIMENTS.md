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

## E3. What a hot-expert map would give (2026-11-15, simulated; built in E4)

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
  Its effect on prompts is unknown. E4 builds it.

## E4. The hot-expert map, built (2026-11-15, measured): 28 % faster decode, the 256K default

The 256K preset's default since 2026-11-16 (`--hot-map` until then); `--no-hot-map` runs the preset without it.

**What it does.** The preset keeps 28 layers' experts whole on the cards. With the map only layers 8 and 75 to 77 stay whole.
The memory of the other 24 holds the busiest experts of every layer instead: 3,584 on card 0 (72 to 156 a layer) and 2,560 on
card 1 (62 to 87). Host memory holds the other 12,032 experts. GPU and host memory are what the preset uses.

**How.** One change to the engine patch, `LLAMA_MOE_HOT_MAP=<file>`:

- **The loader gives a layer's experts a new order,** the pinned ones last, and host memory gets only the others. The router's
  rows take the same order, so the model computes the same function.
- **The expert cache reads the pinned experts from the model file** into the first slots of the layer's group at start (15 to
  17 s) and never evicts them. The other slots work as before.
- **A layer's experts are pinned on its own card,** because a prompt takes cached experts from the layer's card only. So the
  split is not E3's 87 a layer: it is what each card's whole layers free.
- **A path that would read a pinned expert from host memory stops the server.** Engine options this preset does not use and
  that were not checked with pinned experts are refused at start.

The map is `recipes/hotmap-256k.txt`, made by `bench/hotmap.py --pin 3584,2560` from the routing of runs A, B and C of E2.

| | As shipped | With the map |
|---|---|---|
| Decode, median of 24 prompts, tok/s | 26.6 | 34.0 (30.8 to 39.6); 34.5 with the trace on |
| Expert selections computed on a card, all 75 layers | 73.3 % (cache hit rate 57.3 % on 47 layers) | 80.5 % (79.4 % on 71 layers) |
| Prompt to 3.6K / 100K / 201K, tok/s | 849 / 1,494 / 1,276 | 824 / 1,463 / 1,254 |
| First token after those steps, s | 4.2 / 64.9 / 78.7 | 4.4 / 66.2 / 80.1 |
| First token of a fresh 16K / 32K prompt, s | 11.3 / 21.1 | 11.7 / 21.7 |
| First token of a fresh 65K / 130K prompt, s | 42.8 / 88.7 | 43.7 / 90.8 |
| Lowest host memory available, 16 GiB prompt cache full | 27.8 GiB | 27.7 GiB |
| GPU memory at the peak, MiB | 96,513 / 96,270 | 96,489 / 96,268 |

- **Decode is 28 % faster,** twice E3's estimate, with 27 % less expert work on the CPU (26.7 % of the selections to 19.5 %).
  Prompts are 2 to 3 % slower and a first token 2 to 4 % later: the layers that were whole on the cards now copy their
  pinned experts card to card for each prompt batch.
- **It holds on prompts the map has not seen.** A map made from prompts 1 to 12 alone shares 5,341 of its 6,144 experts with
  the full one, and ran prompts 13 to 24 at 34.0 tok/s (34.5 with the full map, 26.4 as shipped). All 24 are bench prompts
  of one kind; code, long conversations and other languages were not tried.
- **The loaded bytes were checked.** `LLAMA_MOE_HOT_VERIFY=1` compares them with the model file at start: 36,096 host expert
  tensors, 36,352 router rows and 18,432 pinned expert tensors, none differ.
- **The model is the same; its rounding is not.** Five prompts of 16 to 9,010 tokens, greedy, 120 tokens, each asked twice
  (`bench/equiv.py`):
  - The shipped engine does not repeat itself: between its two runs the first token of two prompts changed
    and the chosen token's probability moved by up to 0.24 over identical text. With the map it is the same (up to 0.07),
    and the first-token probabilities of the two engines overlap.
  - The first token of one prompt (738 tokens) repeats exactly in each engine and differs between them: 0.160 / 0.148 /
    0.105 for the shipped engine's top three tokens, 0.172 / 0.139 / 0.182 for the same tokens with the map.
  - Two controls place the difference. Other layers whole on the cards (E2's pick) give the shipped numbers exactly. So does
    a test map that pins each layer's last experts and leaves every expert its number. So the pinning is exact, and the new
    order of the experts alone moves the result: the same sums in another order, which this engine amplifies as it does
    between its own runs.
- **Costs.** The start is about 13 s longer (2:06 against 1:53 to the open port, the medians of five starts each). The map is fitted to recorded routing. Only the 256K preset has one. A trace
  still carries the file's expert numbers; `LLAMA_MOE_ROUTE_HIST` carries positions.
- **Not done:** a map from a wider set of prompts. The route bias with the map is E5; maps for the other rungs are E6.

Raw files: `results/experiments/e4-*.json`.

## E5. The route bias with the hot-expert map (2026-11-16, measured): faster, and it moves the output; the default stays 0

`--route-bias V` adds V to the router score of the experts a card already holds, so more tokens pick experts on the cards.
Upstream measured it on its machine only. Here: the 256K preset as shipped and with `--hot-map`, at 0.005, 0.01 and 0.03.

**Speed.** The 24 decode prompts, `scripts/bench.sh`; the map at 0.03 also ran the prompt steps and the size probe.

| 256K preset | Route bias | Decode, tok/s | Gain over bias 0 |
|---|---|---|---|
| As shipped | 0 | 26.6 | |
| | 0.03 | 34.5 (28.4 to 40.9) | +30 % |
| `--hot-map` | 0 | 34.0 (30.8 to 39.6) | |
| | 0.005 | 37.7 (31.4 to 41.9) | +11 % |
| | 0.01 | 39.2 (33.8 to 45.4) | +15 % |
| | 0.03 | 44.7 (39.9 to 48.9) | +31 % |

- **The gains multiply:** the map and 0.03 together are 68 % over the preset as shipped.
- **Prompts do not change.** The map at 0.03: 824 / 1,462 / 1,257 tok/s to 3.6K / 100K / 201K (824 / 1,463 / 1,254 at bias 0),
  first token of a fresh 16K / 32K / 130K prompt after 11.7 / 21.8 / 90.7 s (11.7 / 21.7 / 90.8). The bias only acts on
  batches of up to 32 tokens. Host and GPU memory peaks are the same.

**What it costs.** `llama-perplexity` in its KL-divergence mode on the wikitext-2 test text, 8 chunks of 4,096 tokens, the
second half of each scored (16,376 positions), with each preset's exact environment (`bench/kl.sh`). It runs 4-token batches,
the size decode runs, because the cache and the bias serve only small batches. Upstream's reference, every expert in host
memory, does not fit in 256 GB, so the reference is the map at bias 0, and a second run of it shows the engine's own spread.

| Against the map at bias 0 | Same top token | Mean KL divergence | Perplexity |
|---|---|---|---|
| The map at bias 0, run again | 98.25 ± 0.10 % | 0.0039 | × 1.0012 |
| As shipped, bias 0 | 96.51 ± 0.14 % | 0.0095 | × 1.0008 |
| The map at 0.005 | 96.35 ± 0.15 % | 0.0122 | × 1.0027 |
| The map at 0.01 | 95.75 ± 0.16 % | 0.0154 | × 1.0021 |
| The map at 0.03 | 94.75 ± 0.17 % | 0.0247 | × 1.0034 |
| As shipped, 0.03 | 94.97 ± 0.17 % | 0.0240 | × 1.0018 |

- **The bias moves the output more than anything else here.** Against the map's own spread (98.25 %), 0.005 / 0.01 / 0.03
  change the top token at 1.9 / 2.5 / 3.5 more positions in 100. Perplexity barely sees it (+0.3 % at 0.03).
- **No value passes upstream's rule** for a default (same top token within 0.5 points of bias 0, perplexity within 1 %, 5 %
  more decode), so the default stays 0, with or without the map. 0.03 stays the speed option for anyone who accepts that.
- **The map does not make the bias costlier.** As shipped and with the map, 0.03 lands as far from the reference (94.97 and
  94.75 %, KL 0.024 and 0.025). Read upstream's way, the shipped preset loses 1.54 points from bias 0 to 0.03 against a common
  reference; upstream's two-card presets lost 1.08 to 1.59.
- **A decode run does not repeat exactly, even at bias 0** (1.75 points between two runs of the same preset). Which experts
  sit in the cache at a token differs between runs, and the CPU and the cards round differently. That is also why the shipped
  preset at bias 0 is 3.5 points from the map at bias 0: they compute different experts on the CPU. Upstream's bias-0 presets
  were 2.7 to 3.0 points from its reference for the same reason. Neither is a different model; E4's byte checks stand.
- **Not measured:** a reference outside the engine's decode path (it would need the 384 GB host), more than one run of each
  biased value, values between 0.005 and 0.01 with a longer text, and the bias at 512K to 1M.

Raw files: `results/experiments/e5-*.json` (speed) and `e5-kl.txt` (the KL statistics of every run).

## E6. Hot-expert maps for 512K, 768K and 1M (2026-11-17, measured): 7 to 18 % faster decode, the default on every rung

Each rung's map is made as 256K's (E4): only layers 8 and 75 to 77 stay whole, and the memory of the rung's other whole layers
pins the busiest experts of the card's layers (`bench/hotmap.py --pin N0,N1` from the same traces A, B and C). Host and GPU memory
hold what the preset holds. Each smaller map is a subset of the larger ones. Recipes: `recipes/2gpu-<ctx>-ram256-hot.sh`.

| Rung | Pinned, card 0 / card 1 | Decode, tok/s | Deepest prompt step, tok/s | First token of a fresh 16K / 32K prompt, s | Decode after the deepest step, tok/s |
|---|---|---|---|---|---|
| 256K (E4) | 3,584 / 2,560 | 26.6 -> 34.0 | 1,276 -> 1,254 (201K) | 11.3 / 21.1 -> 11.7 / 21.7 | 27.3 -> 32.1 |
| 512K | 3,072 / 2,048 | 26.7 -> 31.5 | 1,052 -> 1,039 (401K) | 11.8 / 21.7 -> 12.1 / 22.4 | 24.5 -> 30.7 |
| 768K | 2,816 / 1,792 | 27.5 -> 29.3 | 718 -> 715 (602K) | 14.5 / 27.9 -> 15.0 / 28.8 | 22.3 -> 23.3 |
| 1M | 2,560 / 1,536 | 25.6 -> 29.5 | 616 -> 607 (803K) | 20.5 / 41.1 -> 20.8 / 41.7 | 18.3 -> 25.1 |

Without the map, then with it; one run of each.

- **Decode gains less as the rung grows,** because each rung up has fewer whole layers to turn into pins: the maps take an
  estimated 51, 45, 42 and 38 % of the expert traffic (`hotmap.py --holdout`). 768K gains least: 23 of the 24 prompts were faster,
  by 0 to 24 %. Why 768K falls below 1M was not traced.
- **Prompts cost 1 to 3 %,** as at 256K. "Decode after" is one request of about 200 tokens, so it is noisy; 1M's +37 % there is one
  sample.
- **Memory.** GPU peaks are within 70 MiB of the presets'. The lowest host memory available was 18.5, 13.1 and 8.4 GiB, with each
  rung's prompt cache in use but not checked full; without the map the same runs, with a 2 GiB cache, left 25.8, 15.8 and 9.7.
  Host memory holds the same experts with and without the map, and at 256K, with the cache full, the two were 0.1 GiB apart.
- **The engine needed one change.** E4 refused to start unless a card kept 256 of its own cache slots free beside the pins: enough
  for every expert of a layer in one lookup, when the slots borrowed from the compute buffer and lent by the KV cache are gone. The
  larger rungs keep 106 to 313. The start now asks for 8, the experts of one token, as without a map, and each lookup checks that
  its unpinned experts fit the free slots; one that does not stops the server. The case that needs the most is a short prompt
  (under 513 new tokens) deep in a long conversation. At 1M, card 0 with 106, follow-ups of 53, 194 and 376 new tokens after
  748,144 tokens of context all ran (2.9 to 5.4 s to the answer). Prompts of 377 to 512 new tokens at that depth were not tried.

Raw files: `results/experiments/e6-*.json`.

## What is left

1. **A map from wider traffic** than the 24 bench prompts: code, long conversations, other languages.
