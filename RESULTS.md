# Results and method

Every speed figure in this file outside [A 256 GB host](#a-256-gb-host) is upstream's
([killy-netsphere/glm53-offload](https://github.com/killy-netsphere/glm53-offload)), measured on the machine below with the image
this repo builds and the scripts in it. The README's figures are this fork's, from that section. The raw files are in `results/`:
three JSON files per preset (as it ships, and two runs with the route bias), each holding every request with its text and its
timestamps. The tables between the `T:` markers are written from those files by
`bench/report.py --tables`.

## The machine

| | |
|---|---|
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), PCIe 5.0 x16 each, peer-to-peer OK, stock clocks |
| CPU | AMD Ryzen Threadripper PRO 9985WX, 64 cores (one NUMA node) |
| RAM | 768 GB, 8 × 96 GB DDR5-6400 RDIMM |
| OS / driver | Ubuntu 24.04.5, NVIDIA 610.43.02, Docker 29.6.1, NVIDIA Container Toolkit 1.19.1 |
| Model | `UD-Q3_K_XL-a5` (`scripts/get-model.sh`): 336.7 GB in 9 shards, from Unsloth's `UD-Q3_K_XL` at revision `346b3591` |

## Method

One server per preset, started with `./run.sh` as it ships, on whole cards, serving one request at a time. `scripts/bench.sh` runs
two things against it:

- **Decode.** One warmup request, then the 24 prompts of `bench/prompts-24.txt`, each answered once with 512 generated tokens.
  Sampling is greedy (temperature 0) and the MTP head drafts 2 tokens a step. The reasoning effort is the template's default (max):
  GLM-5.3 always thinks, so most of the 512 tokens are reasoning. A request's rate covers the decode phase only: the tokens after the
  first, over the time from the first streamed token to the last. A preset's figure is the median of its 24 prompts, and the table
  gives the lowest and the highest prompt beside it.
- **Prompt.** One prompt grown in three steps: to 4,096 target tokens, then to 7/16 and to 7/8 of the context (for `1gpu-1m`, of
  1,048,576). The filler text counts about 12 % short of its target, and the table gives the real token counts. Each step extends
  the prompt of the step before, so the server computes only the new tokens. The prompt rate is the server's own
  `timings.prompt_per_second` over those tokens. After each step the server generates 256 tokens. Their rate is decode at that
  depth, right after a long prompt. It is lower than the 24-prompt figure: the context is long by then, and a long prompt takes
  part of the memory the expert cache sits on, so the hot experts come back over the next tokens.

"First token after" is the time from the request to the first streamed token, as the client sees it.

The route bias column is the same 24 decode prompts against the same preset started with `--route-bias 0.03`
(`scripts/bench.sh --decode-only`, `results/<preset>-bias003.json`). The gain is the ratio of the two medians. A second server
with the bias ran `scripts/bench.sh` in full, for the prompt steps with it (`results/<preset>-bias003-full.json`); its 24-prompt
medians are within 0.7 % of the first.

## Decode

<!-- T:decode -->
| Preset | Decode, tok/s | Lowest prompt | Highest prompt | First token, s | With `--route-bias 0.03` | Gain |
|---|---|---|---|---|---|---|
| `1gpu-256k` | 38.9 | 35.9 | 45.5 | 0.62 | 51.1 | +31.3 % |
| `1gpu-512k` | 38.1 | 34.4 | 45.4 | 0.61 | 50.4 | +32.2 % |
| `1gpu-1m` | 38.1 | 35.3 | 43.4 | 0.82 | 49.5 | +29.8 % |
| `2gpu-256k` | 61.6 | 55.0 | 70.6 | 0.56 | 75.9 | +23.3 % |
| `2gpu-512k` | 60.2 | 53.8 | 69.3 | 0.55 | 76.0 | +26.3 % |
| `2gpu-1m` | 58.4 | 52.0 | 66.7 | 0.54 | 71.6 | +22.5 % |
<!-- /T:decode -->

## Prompts

<!-- T:prompt -->
| Preset | Prompt grown to | Prompt, tok/s | First token after, s | Decode after it, tok/s |
|---|---|---|---|---|
| `1gpu-256k` | 3,608 | 674 | 5.4 | 34.9 |
|  | 100,376 | 1,624 | 59.7 | 37.4 |
|  | 200,728 | 1,319 | 76.2 | 35.5 |
| `1gpu-512k` | 3,608 | 641 | 5.6 | 33.6 |
|  | 200,728 | 1,489 | 132.5 | 31.3 |
|  | 401,432 | 1,114 | 180.3 | 31.5 |
| `1gpu-1m` | 3,608 | 612 | 5.9 | 32.7 |
|  | 401,432 | 670 | 594.2 | 29.7 |
|  | 802,840 | 545 | 736.8 | 28.4 |
| `2gpu-256k` | 3,608 | 1,068 | 3.4 | 53.7 |
|  | 100,376 | 2,285 | 42.4 | 52.5 |
|  | 200,728 | 1,790 | 56.1 | 46.4 |
| `2gpu-512k` | 3,608 | 664 | 5.4 | 50.7 |
|  | 200,728 | 2,174 | 90.8 | 45.4 |
|  | 401,432 | 1,542 | 130.3 | 41.4 |
| `2gpu-1m` | 3,608 | 660 | 5.5 | 50.0 |
|  | 401,432 | 1,520 | 262.0 | 44.6 |
|  | 802,840 | 949 | 423.1 | 41.4 |
<!-- /T:prompt -->

**With the route bias.** The same three steps, against each preset started with `--route-bias 0.03`.

<!-- T:prompt-bias -->
| Preset | Prompt grown to | Prompt, tok/s: bias 0 | bias 0.03 | First token after, s: bias 0 | bias 0.03 | Decode after it, tokens 65 to 128, tok/s: bias 0 | bias 0.03 |
|---|---|---|---|---|---|---|---|
| `1gpu-256k` | 3,608 | 674 | 673 | 5.4 | 5.4 | 41.6 | 52.1 |
|  | 100,376 | 1,624 | 1,623 | 59.7 | 59.7 | 42.6 | 48.3 |
|  | 200,728 | 1,319 | 1,318 | 76.2 | 76.2 | 39.3 | 52.3 |
| `1gpu-512k` | 3,608 | 641 | 641 | 5.6 | 5.6 | 42.4 | 55.4 |
|  | 200,728 | 1,489 | 1,487 | 132.5 | 132.7 | 38.2 | 51.1 |
|  | 401,432 | 1,114 | 1,114 | 180.3 | 180.3 | 36.3 | 42.4 |
| `1gpu-1m` | 3,608 | 612 | 613 | 5.9 | 5.9 | 42.3 | 55.8 |
|  | 401,432 | 670 | 676 | 594.2 | 589.0 | 35.1 | 43.6 |
|  | 802,840 | 545 | 549 | 736.8 | 731.4 | 28.0 | 34.7 |
| `2gpu-256k` | 3,608 | 1,068 | 1,066 | 3.4 | 3.4 | 66.7 | 84.4 |
|  | 100,376 | 2,285 | 2,281 | 42.4 | 42.5 | 60.3 | 77.4 |
|  | 200,728 | 1,790 | 1,779 | 56.1 | 56.5 | 54.0 | 67.9 |
| `2gpu-512k` | 3,608 | 664 | 665 | 5.4 | 5.4 | 65.9 | 80.7 |
|  | 200,728 | 2,174 | 2,177 | 90.8 | 90.7 | 59.5 | 66.3 |
|  | 401,432 | 1,542 | 1,544 | 130.3 | 130.1 | 45.0 | 59.4 |
| `2gpu-1m` | 3,608 | 660 | 659 | 5.5 | 5.5 | 64.7 | 73.4 |
|  | 401,432 | 1,520 | 1,520 | 262.0 | 262.0 | 50.5 | 60.9 |
|  | 802,840 | 949 | 950 | 423.1 | 422.8 | 46.0 | 49.2 |
<!-- /T:prompt-bias -->

- **The bias does not change prompt speed.** The two settings differ by less than 1 % at every preset and size, in prompt speed
  and in the wait for the first token. The bonus is applied on decode steps; a prompt is routed without it.
- **The decode after a prompt is faster with it,** at every step: by 7 to 34 % over tokens 65 to 128 of the answer, and by 5 to
  19 % over its first 64.
- The last two columns are tokens 65 to 128 of the answer, not the whole of it as in the table above. The answers end at different
  lengths, and the first 64 tokens after a prompt are the slow ones in both settings (the hot experts are coming back to the
  cache), so the whole answer of a shorter reply reads slower. Each step is one answer: read these columns for their direction.
- These answers are about a repetitive text, which the drafts predict well: their rates are not comparable with the 24-prompt
  figures.

## Memory

<!-- T:sizes -->
| Preset | Context, tokens | Expert cache, MiB | Host RAM in use at the peak, GiB |
|---|---|---|---|
| `1gpu-256k` | 262,144 | 30,000 | 339 |
| `1gpu-512k` | 524,288 | 14,400 | 342 |
| `1gpu-1m` | 1,000,000 | 5,400 | 338 |
| `2gpu-256k` | 262,144 | 72,600 | 354 |
| `2gpu-512k` | 524,288 | 50,860 | 343 |
| `2gpu-1m` | 1,048,576 | 59,050 | 342 |
<!-- /T:sizes -->

- **Host.** The last column is the whole host's memory in use (total less available) at the peak of that preset's stability run,
  rounded up. About 10 GiB of it was the test host's own OS and idle services. The peak comes with long prompts and a full context.
  The start refuses a host with less memory available than this figure.
- **GPU.** The presets are sized to fill a whole card. In the stability run the fuller card peaked at 94,196 to 96,734 MiB of its
  97,887.
- **Less RAM.** The presets pin the whole model in host memory. Loading it memory-mapped instead was measured and is not shipped:
  with every page in memory it was 4 to 15 % slower, and it saved no memory. With the container capped at 256 GB, decode fell to
  2.8 and 4.3 tok/s on one card and to 6.1 and 6.2 on two (a story and a code prompt), and at 192 GB to under 3. Two cards on a
  256 GB host have presets that pin less: [A 256 GB host](#a-256-gb-host).

## The route bias

What the bias is, and what the default means, is in the README. This is what it costs.

**Method.** `llama-perplexity` in its KL-divergence mode, on the wikitext-2 test text: 4 chunks of 4,096 tokens, the second half of
each scored (2,047 positions a chunk, 8,188 in all). The reference is the same a5 weights on the same engine at its plain settings:
both cards, a 16-bit KV cache, a 100,000 MiB expert cache with misses computed on the CPU, and no route bias. Each row is a preset's
whole environment against that reference, on its own cards and with the cache size it ships with. "Same top token" is the share of
scored positions where the preset's first choice is the reference's. These rows were measured in the development harness, on the
commit this repo's patch is taken from. `llama-perplexity` holds a 4,096-token context, not the preset's. Where a row has two
figures, the preset was run twice with the bias.

| Preset | Route bias | Same top token | Mean KL divergence | Perplexity, over the reference |
|---|---|---|---|---|
| `1gpu-256k` | 0 | 97.18 ± 0.18 % | 0.0105 | × 1.0040 |
| | 0.03 | 95.08 and 95.16 % | 0.0346 and 0.0344 | × 1.0074 and 1.0075 |
| `1gpu-512k` | 0 | 97.30 ± 0.18 % | 0.0108 | × 1.0033 |
| | 0.03 | 95.08 ± 0.24 % | 0.0353 | × 1.0040 |
| `1gpu-1m` | 0 | 97.03 ± 0.19 % | 0.0109 | × 1.0025 |
| | 0.03 | 94.97 ± 0.24 % | 0.0361 | × 1.0023 |
| `2gpu-256k` | 0 | 97.02 ± 0.19 % | 0.0105 | × 1.0013 |
| | 0.03 | 95.96 and 95.93 % | 0.0245 and 0.0251 | × 1.0049 and 1.0114 |
| `2gpu-512k` | 0 | 97.12 ± 0.19 % | 0.0101 | × 1.0008 |
| | 0.03 | 95.53 ± 0.23 % | 0.0264 | × 1.0048 |
| `2gpu-1m` | 0 | 97.07 ± 0.19 % | 0.0110 | × 1.0025 |
| | 0.03 | 95.76 and 95.79 % | 0.0253 and 0.0250 | × 1.0091 and 1.0078 |

- **Bias 0 is not at 100 %** because a preset differs from the reference in arithmetic, not in routing: an 8-bit KV cache, this
  repo's kernels, and a different split of the experts between the GPU and the CPU (the cache size, the card count, the CPU share).
- **The bias costs 1.1 to 2.2 points of same top token** (2.06, 2.22, 2.06, 1.08, 1.59 and 1.29 in the order of the table, by
  the mean where a preset has two runs) and adds 0.014 to 0.025 to the mean KL divergence. One card loses 2.1 to 2.2 points and
  two cards 1.1 to 1.6; the one-card cache holds fewer of the experts.
- **Perplexity barely sees it.** Every row is within 1.2 % of the reference, and a biased run is within 1.0 % of the bias-0 run
  of its preset; a ratio in this table carries about 0.3 % either way. Perplexity only asks how likely the text's own next token
  is; the top-token and KL columns ask whether the distribution is the same.
- The rule that set the default was fixed before the measurement: 0.03 stays the default only where it keeps same top token within
  0.5 points of bias 0. It did not, on any of the six presets, so the default is 0.
- **A biased run does not repeat exactly.** The three presets run twice with the bias differ by 0.02 to 0.09 points of same top
  token between their two runs; the next section has more on that.

<!-- T:sweep -->
**Values under 0.03.** Two configurations with a 64K context were measured at 0.005, 0.0075, 0.01 and 0.015: the recipes of this
repo for two cards and for one with a 65,536-token context, which leaves the expert cache 108,000 and 38,500 MiB. They are not
shipped as presets.

- **Speed:** the 24 decode prompts, with a bias-0 run of the preset before the biased runs and one after them. The gain is the mean
  over the 24 prompts of the gain of each prompt against those two runs, which differ from each other by 0.1 % on two cards and
  0.3 % on one.
- **Cost:** the method above, against the same reference. "Under bias 0" is the distance to the bias-0 row measured with them.
  An earlier run of the two configurations read 97.07 and 96.96 % at bias 0, and 95.81 and 94.92 % at 0.03 (1.26 and 2.04 points
  under). Five of the eight values were measured twice, and the table gives both runs. A same-top-token figure carries about 0.2
  points either way.
- Both were measured in the development harness, on the commit this repo's patch is taken from. The tok/s column is not from
  `results/`: read it for the differences.

| Preset | Route bias | Decode, tok/s | Gain | Same top token | Under bias 0, points | Mean KL divergence |
|---|---|---|---|---|---|---|
| Two cards, 64K | 0 | 61.9 and 61.9 | | 97.11 % | | 0.0105 |
| | 0.005 | 64.5 | +4.40 % | 96.73 % | 0.379 | 0.0135 |
| | 0.0075 | 67.5 | +8.54 % | 96.52 and 96.59 % | 0.587 and 0.513 | 0.0155 and 0.0144 |
| | 0.01 | 70.1 | +11.62 % | 96.62 and 96.32 % | 0.489 and 0.782 | 0.0158 and 0.0166 |
| | 0.015 | 72.9 | +16.56 % | 96.36 % | 0.745 | 0.0180 |
| One card, 64K | 0 | 39.4 and 39.3 | | 96.92 % | | 0.0110 |
| | 0.005 | 41.9 | +7.85 % | 96.67 and 96.67 % | 0.256 and 0.256 | 0.0146 and 0.0146 |
| | 0.0075 | 42.7 | +9.42 % | 96.64 and 96.54 % | 0.281 and 0.378 | 0.0157 and 0.0163 |
| | 0.01 | 44.0 | +12.60 % | 96.40 and 96.26 % | 0.525 and 0.659 | 0.0175 and 0.0184 |
| | 0.015 | 45.6 | +17.36 % | 95.73 % | 1.197 | 0.0228 |

- **Speed rises in order on both configurations, and the curve bends.** In the same harness 0.03 bought 26 to 27 % on two cards and 30 %
  on one, so half of the bias buys about 60 % of that. All of the gain is time per step: the tokens per step do not move, and the
  cache misses fall, by 33 % at 0.015 on two cards and by 24 % on one.
- **The cost rises with the bias.** The mean KL divergence is in order on both configurations and in both runs, and same top token is in
  order in the means of the two runs. Perplexity does not see it: every biased row is within 0.4 % of its bias-0 row.
- **A biased run does not repeat the way a plain run does.** The two bias-0 runs of each configuration are 0.037 points apart, 3 positions
  of 8,188. The two runs of a biased value are 0.074 and 0.293 points apart on two cards, and 0.000, 0.097 and 0.134 on one. Each
  run is a new process with an empty cache that scores the same text, so this is not the history of a server. Which experts sit in
  the cache at a given token differs a little from run to run, and with a bias that moves the routing. The cause was not traced.
- **The rule,** fixed before the measurement: a value can replace 0 as the default only where it keeps same top token within 0.5
  points of bias 0, perplexity within 1 %, and buys at least 5 % decode. A distance within the run-to-run spread of the 0.5 line
  (0.29 points on two cards, 0.2 on one) counts as at the line, and that is not a pass. A value measured twice is read by its mean.
- **Two cards: no value passes.** 0.005 buys less than 5 %. 0.0075 and 0.01 are at the line (means 0.550 and 0.635), and 0.015 is at
  its far edge or over it.
- **One card: 0.005 passes, and it stays an option.** It gave up 0.256 points in both of its runs, under the 0.3 the rule asks for,
  and bought 7.85 %. It passed by 0.044 points, about 4 positions of 8,188, which is less than the error of the measurement. The
  longer measurement that would settle it (16 chunks, and one deep preset) was not run, so the default stays 0. 0.0075 and 0.01 are
  at the line (means 0.330 and 0.592) and 0.015 is over it.
- **Not measured:** the six presets at these values, and how much an answer depends on what the server handled before it.
<!-- /T:sweep -->

## Stability

Each preset served a seeded mixed load for 40 minutes: short prompts, chats that grow, chats cut back and continued, long prompts,
and answers and prompts cancelled mid-way. Then one prompt filled 96 % of the context. That is 438 to 685 requests a preset and
3,269 in all. Every tenth request was the same fixed prompt, to see whether speed drifts.

- No crash, no engine error, no GPU memory error (ECC counters at 0 before and after) and no Xid.
- Speed held: the last ten fixed prompts of each preset ran within −3.6 % to +5.7 % of its first ten after the cold one.
- No preset ran out of GPU memory. The GPU peaks are in the memory section above.
- The same run covered four more presets, with 64K and 128K contexts, that this release no longer ships, and the read-back was
  done over all ten: 1,003 answers read token by token against a reference server, every fixed prompt and end-of-run answer (703)
  and 30 others a preset (300). 16 positions were flagged, each a token the reference gave under 2 %. None traced to a defect in
  a preset: 12 were near ties between two tokens, 2 were the route bias moving a choice, and at 2 the plain model also gives that
  token 2 to 11 %, so the reference was the outlier.

This run was made with the route bias at 0.03, the setting at the time. The presets differ from it by the default bias of 0. The
speed tables above are from the presets as they ship.

## The requant

The model is Unsloth's `UD-Q3_K_XL` with the tensors read on every token (attention, the shared experts, the dense FFN)
requantized from Q8_0 to Q5_K, using Unsloth's importance matrix. The routed experts, the embedding and the output head are copied
byte for byte. That takes 6.2 GB off what is read for every token (19.5 to 13.3 GB).

| | Unsloth's file | a5 |
|---|---|---|
| Perplexity (wikitext-2 test, 20 chunks of 2,048 tokens) | 2.6530 ± 0.0343 | 2.6574 ± 0.0342 (+0.17 %, inside the error bar) |
| Decode, a story and a code prompt, tok/s | 31.6 / 29.5 | 35.9 / 32.8 (+13.6 % / +11.2 %) |

These two rows are from early in the work: two cards, an earlier engine, the same cache size on both sides. They show what the
requant itself is worth, not what the presets reach. A Q4_K version was 4 to 6 % faster again and cost +1.9 % perplexity, outside
the error bar, so it is not shipped.

## A 256 GB host

The `--ram 256` presets (`recipes/2gpu-ram256.sh` and `recipes/2gpu-*-ram256.sh`) were measured on a second machine, on
2026-11-14. Nothing else in this file is from it. Since 2026-11-16 the 256K preset starts with its hot-expert map; the
`2gpu-256k-ram256` rows below are without it (`--no-hot-map`), and [EXPERIMENTS.md](EXPERIMENTS.md), E4, has it measured.

| | |
|---|---|
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB (97,887 MiB), PCIe 5.0 x16 each, peer-to-peer OK; a desktop session held 113 to 153 MiB of card 0 |
| CPU | AMD Ryzen Threadripper PRO 9955WX, 16 cores (one NUMA node) |
| RAM | 256 GB (246.8 GiB to the OS), 8 GiB of swap |
| Driver / model disk | NVIDIA 615.71.09; the model on a Samsung 9100 PRO NVMe drive |
| Image | this repo's image built at `b5e86ee`, with `recipes/` and `docker/serve.sh` of this change mounted over the image's |

**What the presets change.** The 384 GB presets pin the routed experts of all 75 MoE layers in host memory: 295.8 GiB, more than
this host has. These keep 20 to 28 layers' experts whole on the cards and pin the rest. A card's memory then goes to its weights
(6.2 GiB on card 0, 10.7 on card 1), the resident layers (3,984 MiB each; layer 8 is 5,376 and layers 75 to 77 are 4,872), the KV
cache (12.4 GiB over both cards for each 262,144 tokens), the compute buffer (3.9 to 4.3 GiB), the layer-major staircase (0.5 to
1.0 GiB), and the expert cache in what is left. The upload ring (two slots of the card's largest layer, 10.5 and 9.5 GiB) and part
of the cache share memory with the compute buffer and the KV cache, as in the 384 GB presets. Decode splits the layers across the
cards, as `2gpu-1m` does; the both-cards decode was not tried with resident layers.

| Preset | Context, tokens | Layers on card 0 + card 1 | Host experts, pinned | Expert cache, card 0 / card 1, MiB | Host experts a card caches | Host RAM in use at the peak | GPU peak, card 0 / card 1, MiB |
|---|---|---|---|---|---|---|---|
| `2gpu-256k-ram256` | 262,144 | 15 + 13 | 182.9 GiB | 25,042 / 26,038 | 13.3 / 13.8 % | 204 GiB | 96,513 / 96,270 |
| `2gpu-512k-ram256` | 524,288 | 13 + 11 | 198.4 GiB | 30,022 / 32,014 | 14.7 / 15.7 % | 221 GiB | 94,131 / 95,102 |
| `2gpu-768k-ram256` | 786,432 | 12 + 10 | 206.2 GiB | 34,006 / 35,998 | 16.0 / 17.0 % | 231 GiB | 96,231 / 95,762 |
| `2gpu-1m-ram256` | 1,000,000 | 11 + 9 | 214.0 GiB | 38,986 / 39,982 | 17.7 / 18.2 % | 238 GiB | 96,275 / 95,764 |

- The cache column is its size at the start, with the slots it holds on the compute buffer's and the KV cache's memory. The KV
  cache takes 7 to 25 GiB of it a card back as the context fills.
- Host RAM in use is the total less the available, at the lowest point of the run below; 4 to 7 GiB of it was the host's own OS
  and desktop. Those runs had a 2 GiB prompt cache. The start refuses a host with less available than this with the preset's
  prompt cache full: 219, 227, 233 and 238 GiB ([the prompt cache](#the-prompt-cache-on-the-256-gb-host) below).
- Layers 8 and 75 to 77 are always on the cards. Left on the host they are slot groups of their own in the cache, and with the
  small budget these presets give the cache the engine stopped at the start ("MoE cache budget is too small").

**Method.** One server per preset, started as it ships with one request at a time. A 9,022-token prompt first, then
`scripts/bench.sh` in full (the method at the top of this file; for 768K the steps are 4,096, 344,064 and 688,128 target tokens),
then fresh prompts of eight sizes. Each preset was run once. Raw files: `results/ram256/2gpu-<ctx>.json` and
`results/ram256/2gpu-<ctx>-sizes.json`. `bench/report.py --tables` does not read them; the tables below were written by hand
from those files.

**Decode.**

| Preset | Decode, tok/s | Lowest prompt | Highest prompt | First token, s |
|---|---|---|---|---|
| `2gpu-256k-ram256` | 26.6 | 23.8 | 31.9 | 1.27 |
| `2gpu-512k-ram256` | 26.7 | 22.3 | 32.0 | 1.31 |
| `2gpu-768k-ram256` | 27.5 | 24.5 | 32.4 | 1.28 |
| `2gpu-1m-ram256` | 25.6 | 22.8 | 30.9 | 1.40 |

**One prompt grown in three steps.**

| Preset | Prompt grown to | Prompt, tok/s | First token after, s | Decode after it, tok/s |
|---|---|---|---|---|
| `2gpu-256k-ram256` | 3,608 | 849 | 4.2 | 28.4 |
|  | 100,376 | 1,494 | 64.9 | 28.5 |
|  | 200,728 | 1,276 | 78.7 | 27.3 |
| `2gpu-512k-ram256` | 3,608 | 814 | 4.4 | 29.2 |
|  | 200,728 | 1,344 | 146.7 | 27.7 |
|  | 401,432 | 1,052 | 190.9 | 24.5 |
| `2gpu-768k-ram256` | 3,608 | 794 | 4.5 | 30.7 |
|  | 301,080 | 1,000 | 297.6 | 26.5 |
|  | 602,136 | 718 | 419.7 | 22.3 |
| `2gpu-1m-ram256` | 3,608 | 542 | 6.7 | 28.0 |
|  | 401,432 | 760 | 523.8 | 26.0 |
|  | 802,840 | 616 | 651.8 | 18.3 |

**Fresh prompts by size.** `bench/probe.py --salt`: every prompt starts with a line of its own, so the server computes all of it,
and 64 tokens are generated after it. Three prompts of each size up to 32K, and the table gives their median; one prompt of 64K
and of 128K. The first figure is prompt tokens a second as the server reports them, the second the wait for the first streamed
token as the client sees it.

| Prompt, tokens | `256k`: tok/s, first token | `512k` | `768k` | `1m` |
|---|---|---|---|---|
| 1,049 | 288, 3.6 s | 266, 3.9 s | 255, 4.1 s | 245, 4.2 s |
| 2,058 | 564, 3.6 s | 524, 3.9 s | 503, 4.1 s | 485, 4.2 s |
| 4,101 | 956, 4.3 s | 919, 4.5 s | 898, 4.6 s | 625, 6.6 s |
| 8,161 | 1,246, 6.6 s | 1,180, 6.9 s | 1,040, 7.9 s | 732, 11.2 s |
| 16,309 | 1,460, 11.3 s | 1,403, 11.8 s | 1,139, 14.5 s | 801, 20.5 s |
| 32,606 | 1,571, 21.1 s | 1,525, 21.7 s | 1,184, 27.9 s | 800, 41.1 s |
| 65,197 | 1,538, 42.8 s | 1,497, 43.9 s | 1,161, 56.5 s | 797, 82.2 s |
| 130,354 | 1,470, 88.7 s | 1,441, 90.5 s | 1,124, 116.1 s | 796, 163.8 s |

- **A prompt of 1K to 2K tokens waits about as long as one of 4K:** 3.6 to 4.2 s. From 513 new tokens a prompt takes the
  layer-major graph (`LLAMA_LM_MIN_TOKENS`). That reads like a fixed cost a ubatch; it was not traced.
- **Three prompts of a size agree within 3 %,** except at 768K: 249 to 264 tok/s at 1K, 922 to 1,169 at 8K and 1,138 to 1,207
  at 16K.
- **1M is slower on prompts** than the other three, by 30 to 49 % from 4K tokens up. It runs ubatches of 2K tokens where the
  others run 4K; that this is the cause was not tested.
- The 64 tokens after these prompts ran at 24 to 30 tok/s.

### The prompt cache on the 256 GB host

When a request does not continue the conversation in the slot, the server saves that conversation's state to host memory
(`--cache-ram`) and restores it when a later request returns to it. A state larger than the limit is not saved; otherwise the
oldest entries leave to make room. That is from the server's source at the engine's base commit; the sizes and times below
are measured. The 384 GB presets leave the server's 8,192 MiB. On this host the cache sits next to the pinned experts, so each
preset sets its own (`CACHE_RAM_MIB`), and `--prompt-cache-mib N` overrides it:

| Preset | Prompt cache, MiB | Idle conversations it holds, tokens | Host RAM left of 246.8 GiB, lowest | The start's memory check |
|---|---|---|---|---|
| `2gpu-256k-ram256` | 16,384 | 330K | 27.8 GiB, measured with the bench and the size probe (2026-11-15) | 219 GiB |
| `2gpu-512k-ram256` | 8,192 | 165K | 19.9 GiB: 25.9 measured with a 2 GiB cache, less 6 | 227 GiB |
| `2gpu-768k-ram256` | 4,096 | 83K | 14.7 GiB: 16.7 measured with a 2 GiB cache, less 2 | 233 GiB |
| `2gpu-1m-ram256` | 2,048 | 41K | 9.7 GiB, measured | 238 GiB |

The memory check follows `--prompt-cache-mib`: up by what a larger cache takes, and down by what a smaller one leaves, as far
as the 2 GiB cache every preset was measured with (`256k` with 8,192 MiB: 211 GiB, as measured; with 2,048 or less: 205, and
203.8 was measured).

**A saved conversation is 0.0496 MiB a token** (1 GiB per 20,600 tokens), as the server logged it on `2gpu-256k-ram256`:

| Conversation, tokens | State, MiB |
|---|---|
| 12,042 | 598 |
| 100,042 | 4,966 |
| 150,042 | 7,448 |
| 180,042 | 8,937 |
| 200,042 | 9,929 |

A conversation that fills the 262,144-token context is then 13,012 MiB (12.7 GiB).

**Method.** `bench/switch.py`: conversations of random common words, each ending with the same question, sent in turn to the
one slot with a short unrelated request ("side") between some of them; 16 tokens generated, temperature 0. One run per cache
size, on `2gpu-256k-ram256`. Both were made while the preset had 8,192 MiB, the 16 GiB one with `--prompt-cache-mib 16384`: the
server's arguments are the ones the preset gives it now. "First token" is the client's wall time less the server's generation
time, so it includes saving the conversation that leaves the slot. Raw files: `results/ram256/2gpu-256k-switch-8g.json` and
`2gpu-256k-switch-16g.json`.

| Prompt cache | Step | Tokens computed | Tokens reused | First token, s |
|---|---|---|---|---|
| 8,192 MiB (`--prompt-cache-mib 8192`) | A, 150K, new | 150,027 | 0 | 117.3 |
|  | side | 9 | 10 | 2.1 |
|  | A again | 1 | 150,026 | 0.8 |
|  | B, 12K, new | 12,015 | 12 | 10.6 |
|  | A again | 1 | 150,026 | 0.8 |
|  | B again | 1 | 12,026 | 1.5 |
|  | side | 4 | 15 | 0.3 |
|  | C, 180K, new | 180,017 | 10 | 134.5 |
|  | side: C is not saved | 4 | 15 | 0.5 |
| 16,384 MiB (the preset) | A, 200K, new | 200,027 | 0 | 160.1 |
|  | side | 9 | 10 | 2.6 |
|  | A again | 1 | 200,026 | 1.0 |
|  | B, 100K, new | 100,015 | 12 | 74.4 |
|  | A again | 1 | 200,026 | 1.7 |
|  | B again | 1 | 100,026 | 2.2 |
|  | side | 4 | 15 | 1.3 |

- **8 GiB held the 150K and the 12K conversation together** (8,045 MiB) and refused the 180K one: "prompt state size 8936.717
  MiB exceeds cache size limit 8192.000 MiB, skipping". Returning to C would have cost its 134 s again; that return was not run.
- **16 GiB held the 200K and the 100K conversation together** (14,895 MiB). The server's cache updates took 0.7 to 1.4 s with
  8 GiB and 0.9 to 2.1 s with 16.
- **Memory.** The lowest available host memory was 36.6 GiB with 8 GiB and 30.5 GiB with 16; the cards peaked at 96,493 /
  96,236 and 96,505 / 96,252 MiB, within 20 MiB of the preset's bench run: the prompt cache takes no GPU memory. The 16 GiB
  cache held 14.5 GiB at the most, so full it would leave about 29 GiB. A later run of `scripts/bench.sh` in full and the size
  probe, which fill the cache, left 27.8 GiB: 219.0 GiB in use, the start's memory check.
- **Why the 256K preset has 16 GiB.** It is the smallest round size that holds any one conversation the context allows
  (12.7 GiB) with room for short ones beside it; 8 GiB stops at 165K tokens, and a larger conversation is then computed again
  in full after every switch. It costs 8 GiB of host memory and no GPU memory. Two full conversations need 25.4 GiB, which
  would leave about 19 GiB by arithmetic; that was not run.
- **The answers.** With 16 GiB, the two answers to A after a restore were the same text, and differed from the answer after A
  was computed new; B's two answers were the same. The cause was not traced. The 8 GiB run did not record the answers.
- **Not run:** `512k` and `768k` with their cache full, a cache above 16 GiB, and a save that fails to allocate (the server
  then cuts its limit).

**What was tried and not kept.**

- **The model memory-mapped instead of pinned.** `2gpu-256k` as it ships with `--load-mode mmap`, every expert in the page
  cache: two requests decoded at 1.9 and 4.6 tok/s. `2gpu-768k-ram256` with `--load-mode mmap`: decode 23.8 tok/s over the 24
  prompts (27.5 pinned), the 9,022-token prompt at 96 tok/s (466 pinned), and the 3,608-token step at 208 tok/s with the first
  token after 17.3 s (794 and 4.5 s). That run was stopped there and its JSON was not written; the figures are from its console
  output. So no preset reads the model from the disk after the load, which took 2 to 3 minutes.
- **256K with 24 layers on the cards, 2K-token attention blocks and 16K ubatches:** decode 26.8 tok/s and prompt steps of 769,
  1,246 and 1,125 tok/s, with 17.7 GiB of host memory left at the lowest. The preset has 28 layers, 1K blocks and 4K ubatches:
  the same decode within 1 %, faster prompts, and 43 GiB left.
- **One card** (2026-11-15). The card, the weights, the KV cache and the compute buffer leave room for nine layers' experts at
  256K, and the other 66 layers' are 256.8 GiB: more than the host can pin, so they were memory-mapped, and about 22 GiB more
  than its page cache holds. Tried with the prompt settings of `1gpu-1m` at 262,144 tokens:
  - decode rose from 2.6 tok/s on the first answer to 9 to 10 by the fifth as the page cache settled; the host read 21 GiB
    from the disk during the first 128 tokens and 4 to 7 GiB during later ones;
  - a fresh 1,050-token prompt waited 103 s for its first token (10 tok/s) and a 4,102-token one 105 s (39 tok/s): every batch
    of a prompt reads all the host experts, and the ones that do not fit come from the disk each time. The 64 tokens after
    them ran at 2.2 and 4.6 tok/s;
  - a 16,310-token prompt had no first token after more than 9 minutes, and the trial was stopped there.

  So one card has no preset on this host. A first start with ten layers on the card ran out of GPU memory. A 64K context with
  15 layers on the card, whose host experts (233 GiB) might fit the page cache, was sized and not run. Console output:
  `results/ram256/1gpu-256k-mmap-console.txt`.
- **A smaller CPU share of the cache misses** (`--moe-cache-cpu-share`, 0.9 in every preset), for the 16 cores: on the first
  eight decode prompts of `2gpu-256k-ram256`, 0.5 gave a median of 26.9 tok/s and 0.75 gave 26.6, against 26.3 for the preset on
  the same prompts. One run each, and the spread between two runs was not measured, so the presets keep 0.9.

**What was not measured.** Same top token against the reference, and a stability run (the route bias at 256K is in
[EXPERIMENTS.md](EXPERIMENTS.md), E5): no mixed load, no
cancelled requests, and no prompt past 77 % of the context (80 % for `1m`). `2gpu-512k-ram256` leaves 3.7 and 2.7 GiB of its
cards unused at the peak, which a larger cache could take.

## What was not measured

- **How much an answer depends on the server's history when a bias is set.** With a bias above 0 the routing follows the cache,
  and the cache follows earlier requests. The tables above compare whole runs; they do not isolate this.
- **The reduced-cache path.** When a card is not whole, the start shrinks the expert cache so the server fits. Its speed was not
  measured.
- **More than one request at a time**, other GPUs, other CPUs, and PCIe links under Gen5 x16.
- **Runs longer than the stability run.** Host memory on `1gpu-1m` rose 10.4 GiB in the last 40 minutes of its run; whether that is
  a high-water mark or a slow leak was not settled.
