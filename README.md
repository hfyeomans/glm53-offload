# GLM-5.3 on one or two RTX PRO 6000

Serve **GLM-5.3** (Z.ai's 744B mixture-of-experts model, with DeepSeek-style sparse attention and an MTP head) on **one or two
96 GB RTX PRO 6000 Blackwell** cards. The routed experts stay in host RAM, and the GPUs keep the rest plus a cache of the hot experts.
It is llama.cpp plus one patch: an OpenAI-compatible server, a Docker image, and one command to start it.

| | prompt 3.6K | prompt 25K | prompt 50K | decode |
|---|---|---|---|---|
| **one card** | __ONE_PP__ | | | __ONE_TG__ |
| **two cards** | __TWO_PP__ | | | __TWO_TG__ |

Prompt rates are prompt tokens a second as the server reports them. Decode is generated tokens a second for one user, with MTP
drafting 2 tokens a step, at the default reasoning effort (max), so it is mostly reasoning tokens. Measured on the hardware below; method and every figure's source in [RESULTS.md](RESULTS.md).

## What you need

- **GPUs:** 1 or 2 × RTX PRO 6000 Blackwell (96 GB; the Workstation edition was measured), each on PCIe 5.0 x16. For two cards they
  need peer-to-peer access (`nvidia-smi topo -p2p r` says `OK`). Prompt speed scales with the PCIe link.
- **RAM:** the experts are pinned in host memory: ~325 GiB in use with one card, ~336 GiB with two. 384 GB of RAM is the floor;
  512 GB or more is better.
- **CPU:** it computes most expert-cache misses and part of each prompt, so cores and memory bandwidth matter. Measured on a 64-core
  Threadripper PRO 9985WX with 8 channels of DDR5-6400. The image is built for AVX-512; see `docker/Dockerfile` for an AVX2 build.
- **Software:** Linux, an NVIDIA driver for CUDA 13.0 (measured on 610.43.02), Docker with the NVIDIA Container Toolkit.
- **Disk:** ~681 GB free to prepare the model (download + requant), 337 GB after.

## Quick start

```bash
git clone <this repo> glm53-offload && cd glm53-offload
docker build -t glm53-offload -f docker/Dockerfile .                       # ~5-15 min
scripts/get-model.sh /path/to/models/glm53 --delete-original              # 343 GB download, a ~3 min requant, sha256 checks
./run.sh --model /path/to/models/glm53/UD-Q3_K_XL-a5/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf --gpus 0      # one card
./run.sh --model /path/to/models/glm53/UD-Q3_K_XL-a5/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf --gpus 0,1    # two cards
```

When the log says the server is listening (1-3 min from a warm page cache, longer from disk):

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "messages": [{"role": "user", "content": "Hello"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

GLM-5.3 always thinks: its chat template opens every answer with `<think>`, and `enable_thinking` has no effect. The reasoning arrives
as `reasoning_content`. `reasoning_effort` (`low`, `high`, default `max`) sets how much it thinks.

`./run.sh --help` lists the options:

- `--port`;
- `--bind 0.0.0.0` to serve the LAN (there is no authentication: add `-- --api-key KEY`);
- `--cache-mib`;
- `--threads`;
- `--dry-run` (prints the exact llama-server command and environment).

## How it works

- The model is Unsloth's UD-Q3_K_XL with one change (`scripts/get-model.sh`): the tensors read on every token are requantized from
  Q8_0 to Q5_K with Unsloth's own importance matrix. Those are attention, the shared experts and the dense FFN. Decode is +11-14 % and
  perplexity +0.17 %, inside the error bar. The routed experts are byte-identical to Unsloth's.
- The routed experts of the 78 MoE layers live in pinned host memory. Each GPU holds the attention, the shared experts, the MTP layer,
  the KV cache and an LRU+LFU cache of experts (the rest of its VRAM).
  - Decode: an expert-cache miss is mostly computed on the CPU.
  - Prompts: the experts a ubatch needs are streamed over PCIe, overlapped with compute. The rarely-routed ones are computed on the
    CPU, and the next layer's experts are fetched during the current layer's attention.
- With two cards, prompts of 8K+ tokens run "layer-major": each layer's experts go up once per 16K tokens, and both cards are
  pipelined. Shorter prompts put both cards on every layer.
- Kernels for this model on sm_120:
  - all 64 query heads of the single MLA KV head in one flash-attention tile;
  - a dedicated sparse-attention kernel for the indexer's 2,048 keys per query;
  - a tensor-core lightning indexer;
  - a fused IQ3_XXS gate/up/GLU expert kernel.
- The MTP head drafts 2 tokens a step, with its own sparse attention.

The patch is `patches/glm53-offload.patch` (one diff against upstream `d02b4c0d`). `patches/series/` is the same change as the commit
history, with the authors and messages. Every engine setting is an environment variable set in `recipes/`.

## Limits

- 64K context is the measured preset. The patch has run 512K, but no preset for it ships here yet.
- One request at a time (`--parallel 1`).
- The first prompts after start are slower while the expert cache fills.
- Decode speed follows the CPU's memory bandwidth, and prompt speed follows the PCIe links. Expect different numbers on a different platform, and
  measure with `scripts/bench.sh`.
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

MIT, see [LICENSE](LICENSE). The model weights carry their own license.
