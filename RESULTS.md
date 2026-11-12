# Results and method

Every speed figure in this file and in the README was measured on the machine below, with the image this repo builds and the scripts
in it. The raw files are in `results/`: one JSON per preset, holding every request with its text and its timestamps. The tables
between the `T:` markers are written from those files by `bench/report.py --tables`.

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
(`scripts/bench.sh --decode-only`). The gain is the ratio of the two medians.

## Decode

<!-- T:decode -->
| Preset | Decode, tok/s | Lowest prompt | Highest prompt | First token, s | With `--route-bias 0.03` | Gain |
|---|---|---|---|---|---|---|
| `1gpu-64k` | 39.3 | 34.4 | 45.0 | 0.62 | 51.0 | +29.7 % |
| `1gpu-128k` | 39.9 | 34.7 | 45.8 | 0.62 | 51.0 | +27.7 % |
| `1gpu-256k` | 38.9 | 35.9 | 45.5 | 0.62 | 51.1 | +31.3 % |
| `1gpu-512k` | 38.1 | 34.4 | 45.4 | 0.61 | 50.4 | +32.2 % |
| `1gpu-1m` | 38.1 | 35.3 | 43.4 | 0.82 | 49.5 | +29.8 % |
| `2gpu-64k` | 61.8 | 56.2 | 68.0 | 0.53 | 77.9 | +26.1 % |
| `2gpu-128k` | 62.3 | 55.3 | 72.5 | 0.53 | 77.7 | +24.8 % |
| `2gpu-256k` | 61.6 | 55.0 | 70.6 | 0.56 | 75.9 | +23.3 % |
| `2gpu-512k` | 60.2 | 53.8 | 69.3 | 0.55 | 76.0 | +26.3 % |
| `2gpu-1m` | 58.4 | 52.0 | 66.7 | 0.54 | 71.6 | +22.5 % |
<!-- /T:decode -->

## Prompts

<!-- T:prompt -->
| Preset | Prompt grown to | Prompt, tok/s | First token after, s | Decode after it, tok/s |
|---|---|---|---|---|
| `1gpu-64k` | 3,608 | 869 | 4.2 | 36.4 |
|  | 25,112 | 1,665 | 12.9 | 40.0 |
|  | 50,200 | 1,613 | 15.6 | 40.4 |
| `1gpu-128k` | 3,608 | 858 | 4.2 | 35.2 |
|  | 50,200 | 1,489 | 31.4 | 37.4 |
|  | 100,376 | 1,399 | 35.9 | 37.5 |
| `1gpu-256k` | 3,608 | 674 | 5.4 | 34.9 |
|  | 100,376 | 1,624 | 59.7 | 37.4 |
|  | 200,728 | 1,319 | 76.2 | 35.5 |
| `1gpu-512k` | 3,608 | 641 | 5.6 | 33.6 |
|  | 200,728 | 1,489 | 132.5 | 31.3 |
|  | 401,432 | 1,114 | 180.3 | 31.5 |
| `1gpu-1m` | 3,608 | 612 | 5.9 | 32.7 |
|  | 401,432 | 670 | 594.2 | 29.7 |
|  | 802,840 | 545 | 736.8 | 28.4 |
| `2gpu-64k` | 3,608 | 1,119 | 3.2 | 55.7 |
|  | 25,112 | 1,986 | 10.8 | 59.8 |
|  | 50,200 | 1,884 | 13.3 | 60.7 |
| `2gpu-128k` | 3,608 | 1,105 | 3.3 | 55.7 |
|  | 50,200 | 2,292 | 20.4 | 60.6 |
|  | 100,376 | 2,167 | 23.2 | 56.9 |
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

## Memory

<!-- T:sizes -->
| Preset | Context, tokens | Expert cache, MiB | Host RAM in use at the peak, GiB |
|---|---|---|---|
| `1gpu-64k` | 65,536 | 38,500 | 336 |
| `1gpu-128k` | 131,072 | 29,800 | 339 |
| `1gpu-256k` | 262,144 | 30,000 | 339 |
| `1gpu-512k` | 524,288 | 14,400 | 342 |
| `1gpu-1m` | 1,000,000 | 5,400 | 338 |
| `2gpu-64k` | 65,536 | 108,000 | 345 |
| `2gpu-128k` | 131,072 | 98,000 | 354 |
| `2gpu-256k` | 262,144 | 72,600 | 354 |
| `2gpu-512k` | 524,288 | 50,860 | 343 |
| `2gpu-1m` | 1,048,576 | 59,050 | 342 |
<!-- /T:sizes -->

- **Host.** The last column is the whole host's memory in use (total less available) at the peak of that preset's stability run,
  rounded up. About 10 GiB of it was the test host's own OS and idle services. The peak comes with long prompts and a full context.
  The start refuses a host with less memory available than this figure.
- **GPU.** The presets are sized to fill a whole card. In the stability run the fuller card peaked at 94,196 to 96,836 MiB of its
  97,887 on nine presets. `1gpu-64k` ran that test with one cache step more than it ships with and peaked at 96,874 MiB; with the
  38,500 MiB it ships with, it peaked at 94,438 MiB on the 24 decode prompts.
- **Less RAM.** The presets pin the whole model in host memory. Loading it memory-mapped instead was measured and is not shipped:
  with every page in memory it was 4 to 15 % slower, and it saved no memory. With the container capped at 256 GB, decode fell to
  2.8 and 4.3 tok/s on one card and to 6.1 and 6.2 on two (a story and a code prompt), and at 192 GB to under 3.

## The route bias

What the bias is, and what the default means, is in the README. This is what it costs.

**Method.** `llama-perplexity` in its KL-divergence mode, on the wikitext-2 test text: 4 chunks of 4,096 tokens, the second half of
each scored (2,047 positions a chunk, 8,188 in all). The reference is the same a5 weights on the same engine at its plain settings:
both cards, a 16-bit KV cache, a 100,000 MiB expert cache with misses computed on the CPU, and no route bias. Each row is a preset's
whole environment against that reference, on its own cards and with the cache size it ships with. "Same top token" is the share of
scored positions where the preset's first choice is the reference's. These rows were measured in the development harness, on the
commit this repo's patch is taken from. `llama-perplexity` holds a 4,096-token context, not the preset's.

| Preset | Route bias | Same top token | Mean KL divergence | Perplexity, over the reference |
|---|---|---|---|---|
| `2gpu-64k` | 0 | 97.07 ± 0.19 % | 0.0102 | × 1.0028 |
| | 0.03 | 95.81 ± 0.22 % | 0.0257 | × 1.0094 |
| `2gpu-512k` | 0 | 97.12 ± 0.19 % | 0.0101 | × 1.0008 |
| | 0.03 | 95.53 ± 0.23 % | 0.0264 | × 1.0048 |
| `1gpu-64k` | 0 | 96.96 ± 0.19 % | 0.0114 | × 1.0022 |
| | 0.03 | 94.92 ± 0.24 % | 0.0344 | × 1.0027 |
| `1gpu-512k` | 0 | 97.30 ± 0.18 % | 0.0108 | × 1.0033 |
| | 0.03 | 95.08 ± 0.24 % | 0.0353 | × 1.0040 |
| `1gpu-1m` | 0 | 97.03 ± 0.19 % | 0.0109 | × 1.0025 |
| | 0.03 | 94.97 ± 0.24 % | 0.0361 | × 1.0023 |

- **Bias 0 is not at 100 %** because a preset differs from the reference in arithmetic, not in routing: an 8-bit KV cache, this
  repo's kernels, and a different split of the experts between the GPU and the CPU (the cache size, the card count, the CPU share).
- **The bias costs 1.3 to 2.2 points of same top token** (1.26, 1.59, 2.04, 2.22 and 2.06 in the order of the table) and adds 0.016
  to 0.025 to the mean KL divergence. One card loses 2.0 to 2.2 points and two cards 1.3 to 1.6; the one-card cache holds fewer of
  the experts.
- **Perplexity does not see it.** Every row is within 1 % of the reference, and the two arms of a preset are within 0.7 % of each
  other. Perplexity only asks how likely the text's own next token is; the top-token and KL columns ask whether the distribution
  is the same.
- The rule that set the default was fixed before the measurement: 0.03 stays the default only where it keeps same top token within
  0.5 points of bias 0. It did not, on any preset measured, so the default is 0.
- Five of the ten presets were measured this way. The other five were not.

<!-- T:sweep -->
**Values under 0.03.** The two 64K presets were also measured at 0.005, 0.0075, 0.01 and 0.015.

- **Speed:** the 24 decode prompts, with a bias-0 run of the preset before the biased runs and one after them. The gain is the mean
  over the 24 prompts of the gain of each prompt against those two runs, which differ from each other by 0.1 % on two cards and
  0.3 % on one.
- **Cost:** the method above, against the same reference. "Under bias 0" is the distance to the bias-0 row measured with them,
  which is a second run of the bias-0 row above (97.07 and 96.96 % there). Five of the eight values were measured twice, and the
  table gives both runs. A same-top-token figure carries about 0.2 points either way.
- Both were measured in the development harness, on the commit this repo's patch is taken from. The tok/s column is not from
  `results/`: read it for the differences.

| Preset | Route bias | Decode, tok/s | Gain | Same top token | Under bias 0, points | Mean KL divergence |
|---|---|---|---|---|---|---|
| `2gpu-64k` | 0 | 61.9 and 61.9 | | 97.11 % | | 0.0105 |
| | 0.005 | 64.5 | +4.40 % | 96.73 % | 0.379 | 0.0135 |
| | 0.0075 | 67.5 | +8.54 % | 96.52 and 96.59 % | 0.587 and 0.513 | 0.0155 and 0.0144 |
| | 0.01 | 70.1 | +11.62 % | 96.62 and 96.32 % | 0.489 and 0.782 | 0.0158 and 0.0166 |
| | 0.015 | 72.9 | +16.56 % | 96.36 % | 0.745 | 0.0180 |
| `1gpu-64k` | 0 | 39.4 and 39.3 | | 96.92 % | | 0.0110 |
| | 0.005 | 41.9 | +7.85 % | 96.67 and 96.67 % | 0.256 and 0.256 | 0.0146 and 0.0146 |
| | 0.0075 | 42.7 | +9.42 % | 96.64 and 96.54 % | 0.281 and 0.378 | 0.0157 and 0.0163 |
| | 0.01 | 44.0 | +12.60 % | 96.40 and 96.26 % | 0.525 and 0.659 | 0.0175 and 0.0184 |
| | 0.015 | 45.6 | +17.36 % | 95.73 % | 1.197 | 0.0228 |

- **Speed rises in order on both presets, and the curve bends.** In the same harness 0.03 bought 26 to 27 % on two cards and 30 %
  on one, so half of the bias buys about 60 % of that. All of the gain is time per step: the tokens per step do not move, and the
  cache misses fall, by 33 % at 0.015 on two cards and by 24 % on one.
- **The cost rises with the bias.** The mean KL divergence is in order on both presets and in both runs, and same top token is in
  order in the means of the two runs. Perplexity does not see it: every biased row is within 0.4 % of its bias-0 row.
- **A biased run does not repeat the way a plain run does.** The two bias-0 runs of each preset are 0.037 points apart, 3 positions
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
- **Not measured:** the other eight presets at these values, and how much an answer depends on what the server handled before it.
<!-- /T:sweep -->

## Stability

Each preset served a seeded mixed load, 60 minutes on the two 64K presets and 40 on the other eight: short prompts, chats that
grow, chats cut back and continued, long prompts, and answers and prompts cancelled mid-way. Then one prompt filled 96 % of the
context. That is 438 to 1,245 requests a preset and 6,691 in all. Every tenth request was the same fixed prompt, to see whether
speed drifts.

- No crash, no engine error, no GPU memory error (ECC counters at 0 before and after) and no Xid.
- Speed held: the last ten fixed prompts of each preset ran within −4.6 % to +5.7 % of its first ten after the cold one.
- No preset ran out of GPU memory. The GPU peaks are in the memory section above.
- 1,003 of the answers were then read back token by token against a reference server: every fixed prompt and end-of-run answer
  (703) and 30 others a preset (300). 16 positions were flagged, each a token the reference gave under 2 %. None traced to a
  defect in a preset: 12 were near ties between two tokens, 2 were the route bias moving a choice, and at 2 the plain model also
  gives that token 2 to 11 %, so the reference was the outlier.

This run was made with the route bias at 0.03, the setting at the time. The presets differ from it by the default bias of 0
(and `1gpu-64k` by its cache step). The speed tables above are from the presets as they ship.

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

## What was not measured

- **How much an answer depends on the server's history when a bias is set.** With a bias above 0 the routing follows the cache,
  and the cache follows earlier requests. The tables above compare whole runs; they do not isolate this.
- **The reduced-cache path.** When a card is not whole, the start shrinks the expert cache so the server fits. Its speed was not
  measured.
- **More than one request at a time**, other GPUs, other CPUs, and PCIe links under Gen5 x16.
- **Runs longer than the stability run.** Host memory on `1gpu-1m` rose 10.4 GiB in the last 40 minutes of its run; whether that is
  a high-water mark or a slow leak was not settled.
