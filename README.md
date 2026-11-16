# GLM-5.3 on two RTX PRO 6000 and 256 GB of RAM

> **A fork of [killy-netsphere/glm53-offload](https://github.com/killy-netsphere/glm53-offload) by Killy (NetSphere).** The engine
> patch, the kernels, the requantized model, the Docker image, the benchmark method and the original presets are that project's
> work. This fork adds presets for a host with less memory: [What this fork changes](#what-this-fork-changes).

Serve **GLM-5.3** (Z.ai's 744B mixture-of-experts model, with DeepSeek-style sparse attention and an MTP head) on **two 96 GB
RTX PRO 6000 Blackwell** cards and a host with **256 GB of RAM**, at 256K to 1M tokens of context. It is llama.cpp plus one
patch: an OpenAI-compatible server, a Docker image, and one command to start it.

| Context | `--ctx` | Decode, tok/s | Prompt, tok/s (deepest step) | First token of a fresh 16K / 32K prompt | Prompt cache | Host RAM free at the lowest |
|---|---|---|---|---|---|---|
| 256K (default) | `256k` | 26.6 | 1,276 (to 201K) | 11.3 s / 21.1 s | 16 GiB | 28 GiB |
| 512K | `512k` | 26.7 | 1,052 (to 401K) | 11.8 s / 21.7 s | 8 GiB | about 20 GiB |
| 768K | `768k` | 27.5 | 718 (to 602K) | 14.5 s / 27.9 s | 4 GiB | about 15 GiB |
| 1M | `1m` | 25.6 | 616 (to 803K) | 20.5 s / 41.1 s | 2 GiB | 10 GiB |

- **Decode** is generated tokens a second for one user: the median over 24 prompts, 512 tokens each, with the MTP head drafting
  2 tokens a step. GLM-5.3 always thinks, so most of those tokens are reasoning.
- **Prompt** is prompt tokens a second as the server reports them, while one prompt grows to that many tokens.
- **Measured** on a 16-core Threadripper PRO 9955WX with 256 GB and two Max-Q cards, one run a preset, with `scripts/bench.sh`.
  The method, every step and the raw files are in [RESULTS.md](RESULTS.md), "A 256 GB host".
- **Host RAM free** is with the preset's prompt cache full. 256K and 1M were measured; the other two are a measured run plus
  arithmetic for the cache.

## What this fork changes

Upstream pins the routed experts of all 75 MoE layers in host memory: 296 GiB, so it needs 384 GB of RAM. A 256 GB host cannot.

- **A 256 GB tier.** `--ram 256`, which the start picks by itself below 360 GiB of RAM. Its presets keep the experts of 20 to
  28 layers whole on the cards and pin the rest in host memory (183 to 214 GiB). A layer on the cards never misses; the expert
  cache serves the others.
- **A four-rung context ladder:** 256K, 512K, 768K (new) and 1M. Each rung up gives two to four layers' worth of GPU memory to
  the KV cache, so more of the experts moves to the host.
- **A prompt-cache profile per rung:** 16, 8, 4 and 2 GiB, what the host has left at that context. `--prompt-cache-mib N`
  overrides it, and the start's memory check rises and falls with it.
- **Benchmarks:** `scripts/bench.sh` knows the 768K context, `bench/switch.py` measures the prompt cache, and the raw files
  are in `results/ram256/`.
- **A hot-expert map for 256K,** `--hot-map`, off by default. Each card pins the busiest experts of all its layers
  instead of some layers whole, so the CPU computes 27 % less of the expert work: decode goes from 26.6 to 34.0 tok/s on
  the same GPU and host memory. The model is the same: its experts are reordered, not changed. Cost: prompts are 2 to 3 %
  slower (first token of a fresh 16K / 32K prompt after 11.7 / 21.7 s), and the start takes about 13 s more. Trade-off:
  the map is fitted to recorded routing, and only this rung has one. It is one change to the engine patch;
  [EXPERIMENTS.md](EXPERIMENTS.md), E4, has the method and the checks.
- **Not changed:** the image recipe and the model. Upstream's presets for a 384 GB host are still in `recipes/`
  and were not run here; [upstream's README](https://github.com/killy-netsphere/glm53-offload#readme) has them.

## What you need

- **GPUs:** 2 × RTX PRO 6000 Blackwell (96 GB; the Max-Q Workstation edition was measured), each on PCIe 5.0 x16, with
  peer-to-peer access (`nvidia-smi topo -p2p r` says `OK`) and nothing else on them: the presets peaked at 94,131 to
  96,513 MiB of a card's 97,887. A desktop session of about 150 MiB on card 0 is allowed for.
- **RAM:** 256 GB (246.8 GiB to the OS). The start refuses with less than 219, 227, 233 or 238 GiB available for `256k`,
  `512k`, `768k` or `1m`. Pinned memory cannot be swapped out, so close what else runs on the host for the larger rungs.
- **CPU:** it computes most expert-cache misses, so cores and memory bandwidth matter. The image is built for AVX-512; see
  `docker/Dockerfile` for an AVX2 build.
- **Software:** Linux, an NVIDIA driver for CUDA 13.0 (measured on 615.71.09), Docker with the NVIDIA Container Toolkit.
- **Disk:** 337 GB for the model. The load took 2 to 3 minutes from an NVMe drive.

## Quick start

```bash
git clone https://github.com/hfyeomans/glm53-offload && cd glm53-offload
docker build -t glm53-offload -f docker/Dockerfile .                       # ~5-15 min
scripts/get-model.sh /path/to/models/glm53                                 # 337 GB download, sha256 checks
./run.sh --model /path/to/models/glm53/UD-Q3_K_XL-a5/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf --gpus 0,1
```

The start checks the machine first (GPUs, peer-to-peer, host memory, the locked-memory limit) and says what is missing. When
the log says the server is listening:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "messages": [{"role": "user", "content": "Hello"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

GLM-5.3 always thinks: `enable_thinking` has no effect, and the reasoning arrives as `reasoning_content`. `reasoning_effort`
(`low`, `high`, default `max`) sets how much it thinks.

`./run.sh --help` lists the options:

- `--ctx 256k|512k|768k|1m`: the rung (default `256k`);
- `--prompt-cache-mib N`: host memory for saved conversations (default: the rung's);
- `--port`, and `--bind 0.0.0.0` to serve the LAN (there is no authentication: add `-- --api-key KEY`);
- `--detach` to run in the background;
- `--route-bias 0.03`: upstream's speed option (default 0, see [below](#the-route-bias));
- `--ram 256|384`, `--cache-mib`, `--threads`;
- `--dry-run` (prints the exact llama-server command and environment, and starts nothing).

## Choosing a rung

Take the smallest context your work fits in.

- **256K is the fastest on prompts and leaves the most host memory.** Decode is 25.6 to 27.5 tok/s on every rung.
- **Any prompt of more than 512 new tokens waits at least 3.6 to 4.3 s** for its first token (6.6 s from 4K tokens at 1M):
  each batch of a prompt uploads the experts of every host layer once. A short prompt waits 1.3 to 1.4 s.
- **1M takes the whole machine.** It is 1,000,000 tokens, and the host was left 10 GiB at the lowest: use `768k` if anything
  else runs on the host.

| | `256k` | `512k` | `768k` | `1m` |
|---|---|---|---|---|
| Layers' experts on the cards | 28 | 24 | 22 | 20 |
| Experts pinned in host RAM | 183 GiB | 198 GiB | 206 GiB | 214 GiB |

## The prompt cache

When a request does not continue the conversation in the slot, the server saves that conversation to host memory, 1 GiB per
20,600 tokens, and restores it when a request returns to it. A conversation larger than the cache is not saved and is computed
again in full.

- **256K has 16 GiB:** enough for any one conversation the context allows (12.7 GiB at 262,144 tokens) and short ones beside
  it. Measured: a 200K-token conversation that took 160 s to compute came back in 1.0 to 1.7 s after another request.
- **The larger rungs have 8, 4 and 2 GiB**, about 165K, 83K and 41K tokens of idle conversations. The host has no more to give
  at those contexts. Their sizes are arithmetic: only 256K was run with its cache full.
- **It matters when requests interleave** on the one slot: an agent with subagents, or side requests such as title generation.

## The route bias

A token whose expert is not on a GPU waits for the CPU or for an upload. `--route-bias` adds a small bonus to the router score
of the experts the cache already holds, so more tokens find theirs on the GPU.

- **The default is 0:** the router picks exactly the experts the model defines. Every figure above is at 0.
- **On this host, 256K, it gives about 30 % more decode:** 26.6 to 34.5 tok/s as shipped, and 34.0 to 44.7 with `--hot-map`.
  Prompt speed and first-token times do not change.
- **It changes the output.** Against the `--hot-map` preset's own output at bias 0, 0.03 changes the top token at 3.5 more positions in
  100, and 0.005 and 0.01 at 1.9 and 2.5 (wikitext-2, `bench/kl.sh`). Upstream's rule for a default allows 0.5, so the
  default stays 0. With any bias above 0, an answer can also depend on what the server handled before.
- [EXPERIMENTS.md](EXPERIMENTS.md), E5, has the method and every run. Values above 0.03 are refused without `--force`.

## How it works

- **The model** is Unsloth's UD-Q3_K_XL with one change by upstream: the tensors read on every token are requantized from Q8_0
  to Q5_K. The routed experts are byte-identical to Unsloth's. `scripts/get-model.sh` downloads the result.
- **Each card** holds its half of the layers' attention and shared experts, the routed experts of its resident layers, the KV
  cache, and a cache of the other layers' hot experts in what is left. Part of that cache sits on memory the compute buffer and
  the KV cache take back when a long prompt runs and as the context fills.
- **The host** holds the other layers' routed experts in pinned memory.
- **Decode:** the layers are split across the two cards. An expert-cache miss is mostly computed on the CPU.
- **Prompts:** from 513 new tokens they run "layer-major", with both cards pipelined: each layer's experts go up once for a
  block of prompt tokens.
- **The MTP head** drafts 2 tokens a step.

The engine is `patches/glm53-offload.patch`, one diff against llama.cpp `d02b4c0d`, with upstream's kernels for this model on
these cards. Every engine setting is an environment variable set in `recipes/`: `common.sh` holds what all presets share,
`2gpu.sh` what two cards add, `2gpu-ram256.sh` what this host adds, and `2gpu-<ctx>-ram256.sh` the rung.

## Measure it yourself

```bash
scripts/bench.sh            # against the running server; it reads the context from the server
```

It runs the 24 decode prompts, then one prompt grown in three steps with 256 tokens generated at each depth, and writes a JSON
file with every request in it. `bench/switch.py` switches conversations on the slot, for the prompt cache. Python 3, no packages.

## Limits

- **Two cards only, for now.** One card is not supported on a 256 GB host.
- One request at a time (`--parallel 1`).
- The cards must be whole. If a card has less free memory than a whole one, the start makes the expert cache smaller and says
  so; that path's speed was not measured.
- The first prompts after a start are slower while the expert cache fills, and decode right after a long prompt starts lower
  and recovers.
- Each preset was run once. A long mixed load, cancelled requests and prompts past 77 % of the context (80 % for `1m`) were not
  run on this host.
- Tested on RTX PRO 6000 Blackwell only (sm_120). Other GPUs are refused unless `--force`.

## Credits

- **[killy-netsphere/glm53-offload](https://github.com/killy-netsphere/glm53-offload)** by Killy (NetSphere), written with
  Claude (Anthropic): the offload work, the kernels and the measurements this fork starts from.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), and two open pull requests upstream builds on:
  - [#30966](https://github.com/ggml-org/llama.cpp/pull/30966), the MoE expert cache with misses run on the CPU, by Aman Gupta
    (am17an);
  - [#29963](https://github.com/ggml-org/llama.cpp/pull/29963), pipeline parallelism with MoE experts in host RAM, by Roman
    Marchenko (drrros).
- [Unsloth](https://huggingface.co/unsloth/GLM-5.3-GGUF) for the quant and its importance matrix.
- Z.ai for GLM-5.3.

The 256 GB presets and their measurements are by Hank Yeomans (hfyeomans), written with Claude (Anthropic).

## License

MIT, see [LICENSE](LICENSE). The model weights carry their own license, the GLM-5.3 License; its text ships with the weights.
