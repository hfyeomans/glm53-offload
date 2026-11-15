# GLM-5.3 on one or two RTX PRO 6000

Serve **GLM-5.3** (Z.ai's 744B mixture-of-experts model, with DeepSeek-style sparse attention and an MTP head) on **one or two
96 GB RTX PRO 6000 Blackwell** cards, at 256K to 1M tokens of context. The routed experts stay in host RAM, and the GPUs keep the rest
plus a cache of the hot experts. It is llama.cpp plus one patch: an OpenAI-compatible server, a Docker image, and one command to
start it.

<!-- T:headline -->
| 256K context | Decode | Prompt, to 3.6K tokens | to 100K | to 201K |
|---|---|---|---|---|
| **One card** | **38.9 tok/s** | 674 tok/s | 1,624 | 1,319 |
| **Two cards** | **61.6 tok/s** | 1,068 tok/s | 2,285 | 1,790 |
<!-- /T:headline -->

- **Decode** is generated tokens a second for one user: the median over 24 different prompts, 512 tokens each, with the MTP head
  drafting 2 tokens a step. GLM-5.3 always thinks, at reasoning effort max by default, so most of those tokens are reasoning.
- **Prompt** is prompt tokens a second as the server reports them, while one prompt grows to that many tokens.
- Both are at the default settings, on the hardware below, taken with `scripts/bench.sh`. The method, every preset and the raw files
  are in [RESULTS.md](RESULTS.md). A speed option trades a little of the model's output for more decode:
  [the route bias](#the-route-bias).

## What you need

- **GPUs:** 1 or 2 × RTX PRO 6000 Blackwell (96 GB; the Workstation edition was measured), each on PCIe 5.0 x16, with nothing else
  on them: the presets fill a card to within about 0.5 GiB. Two cards need peer-to-peer access (`nvidia-smi topo -p2p r` says
  `OK`). Prompt speed scales with the PCIe link.
- **RAM:** 384 GB or more for the presets below. The whole model is pinned in host memory, and the host peaked at 338 to 354 GiB in
  use, the OS included, depending on the preset. Memory-mapping the weights is no way under that: on a host capped at 256 GB, decode
  fell to 3-6 tokens a second. A host with 256 GB and two cards has presets of its own, which keep part of the experts on the cards:
  [A 256 GB host](#a-256-gb-host).
- **CPU:** it computes most expert-cache misses and part of each prompt, so cores and memory bandwidth matter. Measured on a 64-core
  Threadripper PRO 9985WX with 8 channels of DDR5-6400. The image is built for AVX-512; see `docker/Dockerfile` for an AVX2 build.
- **Software:** Linux, an NVIDIA driver for CUDA 13.0 (measured on 610.43.02), Docker with the NVIDIA Container Toolkit.
- **Disk:** 337 GB for the model.

## Quick start

```bash
git clone https://github.com/killy-netsphere/glm53-offload && cd glm53-offload
docker build -t glm53-offload -f docker/Dockerfile .                       # ~5-15 min
scripts/get-model.sh /path/to/models/glm53                                 # 337 GB download, sha256 checks
./run.sh --model /path/to/models/glm53/UD-Q3_K_XL-a5/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf --gpus 0      # one card
./run.sh --model /path/to/models/glm53/UD-Q3_K_XL-a5/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf --gpus 0,1    # two cards
```

The start checks the machine first (GPUs, peer-to-peer, host memory, the locked-memory limit) and says what is missing. When the log
says the server is listening:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "messages": [{"role": "user", "content": "Hello"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

GLM-5.3 always thinks: its chat template opens every answer with `<think>`, and `enable_thinking` has no effect. The reasoning arrives
as `reasoning_content`. `reasoning_effort` (`low`, `high`, default `max`) sets how much it thinks.

`./run.sh --help` lists the options:

- `--ctx 256k|512k|1m`: the context preset (default `256k`);
- `--ram 256|384`: the host memory tier (default: 256 on a host with less than 360 GiB of RAM, see [A 256 GB host](#a-256-gb-host));
- `--route-bias 0.03`: the speed option (default 0);
- `--port`, and `--bind 0.0.0.0` to serve the LAN (there is no authentication: add `-- --api-key KEY`);
- `--detach` to run in the background;
- `--prompt-cache-mib N`: host memory for conversations saved when another one takes the slot (default: the preset's);
- `--cache-mib`, `--threads`;
- `--dry-run` (prints the exact llama-server command and environment, and starts nothing).

## Context presets

Each context has its own preset for one card and for two: six recipes, each measured as it ships.

<!-- T:presets -->
| Cards | Context | Decode, tok/s | With `--route-bias 0.03` | Prompt, tok/s (deepest step) | Decode at that depth, tok/s | Host RAM in use |
|---|---|---|---|---|---|---|
| 1 | 256K | 38.9 | 51.1 (+31 %) | 1,319 (to 201K) | 35.5 | 339 GiB |
| 1 | 512K | 38.1 | 50.4 (+32 %) | 1,114 (to 401K) | 31.5 | 342 GiB |
| 1 | 1M | 38.1 | 49.5 (+30 %) | 545 (to 803K) | 28.4 | 338 GiB |
| 2 | 256K | 61.6 | 75.9 (+23 %) | 1,790 (to 201K) | 46.4 | 354 GiB |
| 2 | 512K | 60.2 | 76.0 (+26 %) | 1,542 (to 401K) | 41.4 | 343 GiB |
| 2 | 1M | 58.4 | 71.6 (+23 %) | 949 (to 803K) | 41.4 | 342 GiB |
<!-- /T:presets -->

Take the smallest context your work fits in. A larger context takes GPU memory from the expert cache, so fewer experts sit on the
GPU. The one-card 1M preset is 1,000,000 tokens (the server reports 1,000,192, the next multiple of 256) and leaves the cache almost
nothing to give: the card must be whole.

## A 256 GB host

Two cards also serve the model from a host with 256 GB of RAM: `--ram 256`, which the start picks by itself on a host with less
than 360 GiB. These presets keep the routed experts of 20 to 28 of the 75 MoE layers whole on the cards and pin the other layers'
in host memory, 183 to 214 GiB instead of 296. A layer on the cards never misses; the expert cache serves the rest. One card
cannot hold enough of the experts for this.

| Context | `--ctx` | Layers on the cards | Pinned in host RAM | Decode, tok/s | Prompt, tok/s (deepest step) | First token of a fresh 16K / 32K prompt | Host RAM free at the lowest |
|---|---|---|---|---|---|---|---|
| 256K | `256k` | 28 | 183 GiB | 26.6 | 1,276 (to 201K) | 11.3 s / 21.1 s | 43 GiB |
| 512K | `512k` | 24 | 198 GiB | 26.7 | 1,052 (to 401K) | 11.8 s / 21.7 s | 26 GiB |
| 768K | `768k` | 22 | 206 GiB | 27.5 | 718 (to 602K) | 14.5 s / 27.9 s | 17 GiB |
| 1M | `1m` | 20 | 214 GiB | 25.6 | 616 (to 803K) | 20.5 s / 41.1 s | 10 GiB |

- **These are from another machine** than the tables above: a 16-core Threadripper PRO 9955WX with 256 GB and two Max-Q cards.
  Decode is less than half of the 64-core host's two-card figure. Three things differ: a quarter of the cores for the cache
  misses, a cache of 13 to 18 % of the host experts, and layers split across the cards on decode instead of both cards on every
  layer. Which of them costs the most was not separated. One run a preset, with `scripts/bench.sh`. [RESULTS.md](RESULTS.md) has
  the method, every step, and prompt sizes from 1K to 128K.
- **Any prompt of more than 512 new tokens waits at least 3.6 to 4.3 s** for its first token (6.6 s from 4K tokens at 1M): each
  batch of a prompt uploads the experts of every host layer once.
- **1M takes the whole machine.** It is 1,000,000 tokens, as on one card. The host was left with 10 GiB at the lowest, and pinned
  memory cannot be swapped out: close what else runs on the host, or use `768k`. The start refuses when less memory is available
  than the preset used (238 GiB for `1m`; the idle test host had 239).
- **No rung reads the model from the disk.** All four pin their host experts. Memory-mapping them was tried at 768K and was not
  kept: a 3.6K-token prompt ran at 208 tok/s instead of 794.
- **The prompt cache follows the context: 16, 8, 4 and 2 GiB** for `256k`, `512k`, `768k` and `1m`. When a request does not
  continue the conversation in the slot, the server saves that conversation to host memory, 1 GiB per 20,600 tokens, and restores
  it when a request returns to it. So about 330K, 165K, 83K and 41K tokens of idle conversations survive; one larger than the
  cache is not saved and is computed again. The "host RAM free" column was measured with a 2 GiB cache: with the preset's cache
  full, 256K is left about 29 GiB (30.5 measured with 14.5 GiB in it), 512K about 20 and 768K about 15 (arithmetic, not run).
- **256K has 16 GiB** because that holds any one conversation the context allows (12.7 GiB at 262,144 tokens) and short ones
  beside it. Measured: a 200K-token conversation that took 160 s to compute came back in 1.0 to 1.7 s after another request.
  With 8 GiB (`--prompt-cache-mib 8192`, which gives 8 GiB of host memory back) a 150K one came back in 0.8 s and a 180K one was
  not saved. The start's memory check follows the cache: 219 GiB for `256k` as it ships, 211 with 8 GiB, 235 with 32.
- **Not measured here:** the route bias, a long mixed load, and the same-top-token comparison. The presets route as the model
  defines; what differs from the 384 GB presets is where an expert is computed.

## The route bias

GLM-5.3 sends every token through a few of each layer's experts, and only part of the experts fit on the GPU at a time. A token whose
expert is not in the cache waits for the CPU or for an upload. The route bias adds a small bonus to the router score of the experts
the cache already holds, so more tokens find their experts on the GPU. The weights that mix the experts' outputs are not biased; what
changes is which experts get picked.

- **The default is 0.** The router picks exactly the experts the model defines. Every figure above is at 0.
- **`--route-bias 0.03` is the speed option.** It is faster by the amounts in the table above, and its output is no longer the plain
  model's. Against the same weights run plainly, the presets pick the same top token at 97.0 to 97.3 % of positions at bias 0, and at
  95.0 to 95.9 % at 0.03: 1.1 to 2.2 positions in 100 change their first choice. Perplexity moves by 1 % at most.
- **It speeds up decode only.** Prompt speed and the wait for the first token are the same with the bias, within 1 % at every
  preset and prompt size. The decode that follows a long prompt is faster with it.
- **With any bias above 0, an answer can depend on what the server handled before it**, because the bonus follows the cache. How
  much was not measured.

<!-- T:bias -->
**Smaller values** were measured with a 64K context, a configuration this release does not ship as a preset (it leaves the expert
cache more room than any preset here). Speed and cost rise together, and no value is free:

| `--route-bias` | Two cards, 64K: decode gain | First choices changed, in 100 | One card, 64K: decode gain | First choices changed, in 100 |
|---|---|---|---|---|
| 0.005 | +4 % | 0.4 | +8 % | 0.3 |
| 0.0075 | +9 % | 0.5 | +9 % | 0.3 |
| 0.01 | +12 % | 0.6 | +13 % | 0.6 |
| 0.015 | +17 % | 0.7 | +17 % | 1.2 |
| 0.03 | +26 % | 1.3 | +30 % | 2.0 |

- **The curve bends.** Half of the bias (0.015) buys about 60 % of the speed of 0.03, and a sixth of it (0.005) buys 4 to 8 %.
- **The changed column is good to about 0.2, and a biased server does not repeat itself the way a plain one does.** Two runs of the
  same text at bias 0 differed by 0.04. Two runs at the same bias differed by up to 0.3 on two cards and 0.1 on one.
- **No value replaces 0 as the default.** The rule asked for at least 5 % more decode for at most 0.5 changed in 100, clear of that
  spread. On two cards no value met it. On one card 0.005 did (+8 % for 0.3), by less than the measurement error: it is an option,
  not the default.
- The six presets were not measured at these values. `RESULTS.md` has every run.
<!-- /T:bias -->

Values above 0.03 were not measured, and the start refuses them without `--force`.

## How it works

- The model is Unsloth's UD-Q3_K_XL with one change: the tensors read on every token (attention, the shared experts and the dense
  FFN) are requantized from Q8_0 to Q5_K with Unsloth's own importance matrix. Measured early in the work, that was +11 to +14 %
  decode for +0.17 % perplexity, inside the error bar. The routed experts are byte-identical to Unsloth's. `scripts/get-model.sh`
  downloads the result, or builds it from Unsloth's files with `--build`.
- The routed experts of the 78 MoE layers live in pinned host memory. Each GPU holds the attention, the shared experts, the MTP layer,
  the KV cache and a cache of experts in the rest of its memory. Part of that cache sits on memory the compute buffer and the KV cache
  take back when a long prompt runs and as the context fills.
- **Decode:** an expert-cache miss is mostly computed on the CPU. With two cards, both work on every layer: each computes half the
  attention heads and the experts it caches (at 1M the layers are split across the cards instead).
- **Prompts on one card:** the experts a batch needs are streamed over PCIe, overlapped with compute. The rarely-routed ones are
  computed on the CPU, and the next layer's experts are fetched during the current layer's attention. From 256K of context, long
  prompts run "layer-major": each layer's experts go up once for a block of prompt tokens.
- **Prompts on two cards:** long prompts run layer-major with both cards pipelined; shorter ones put both cards on every layer.
- Kernels for this model on sm_120:
  - all 64 query heads of the single MLA KV head in one flash-attention tile;
  - a dedicated sparse-attention kernel for the indexer's 2,048 keys per query;
  - a tensor-core lightning indexer, and a fused q8_0 one;
  - a fused IQ3_XXS gate/up/GLU expert kernel.
- The MTP head drafts 2 tokens a step, with its own sparse attention.

The patch is `patches/glm53-offload.patch` (one diff against upstream `d02b4c0d`). `patches/series/` is the same change as the commit
history, with the authors and messages. Every engine setting is an environment variable set in `recipes/`: `common.sh` holds what
all six presets share, `1gpu.sh` and `2gpu.sh` what the card count adds, and one file per preset the rest. `2gpu-ram256.sh` and
the four `2gpu-*-ram256.sh` files are the presets for a 256 GB host.

## Measure it yourself

```bash
scripts/bench.sh            # against the running server; it reads the context from the server
```

It runs what RESULTS.md ran (the 24 decode prompts, then one prompt grown in three steps with 256 tokens generated at each depth)
and writes a JSON file with every request in it. Python 3, no packages.

## Limits

- One request at a time (`--parallel 1`).
- The cards must be whole. If a card has less free memory than a whole one, the start makes the expert cache smaller and says so;
  that path keeps the server from running out of memory, and its speed was not measured.
- The first prompts after a start are slower while the expert cache fills, and decode right after a long prompt starts lower and
  recovers (RESULTS.md has decode at depth).
- Decode speed follows the CPU's memory bandwidth, and prompt speed follows the PCIe links. Expect different numbers on a different
  platform, and measure with `scripts/bench.sh`.
- Tested on RTX PRO 6000 Blackwell only (sm_120). Other GPUs are refused unless `--force`, and the recipes would not fit them anyway.

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- Two open pull requests this builds on:
  - [#30966](https://github.com/ggml-org/llama.cpp/pull/30966), the MoE expert cache with misses run on the CPU, by Aman Gupta
    (am17an);
  - [#29963](https://github.com/ggml-org/llama.cpp/pull/29963), pipeline parallelism with MoE experts in host RAM, by Roman
    Marchenko (drrros).
- [Unsloth](https://huggingface.co/unsloth/GLM-5.3-GGUF) for the quant and its importance matrix.
- Z.ai for GLM-5.3.

The offload work, the kernels and the measurements are by NetSphere, written with Claude (Anthropic).

## License

MIT, see [LICENSE](LICENSE). The model weights carry their own license, the GLM-5.3 License; its text ships with the weights.
