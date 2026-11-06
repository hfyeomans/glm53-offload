# Results and method

## The machine

| | |
|---|---|
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), PCIe 5.0 x16 each, peer-to-peer OK, stock clocks |
| CPU | AMD Ryzen Threadripper PRO 9985WX, 64 cores (one NUMA node) |
| RAM | 768 GB, 8 × 96 GB DDR5-6400 RDIMM |
| OS / driver | Ubuntu 24.04.5, NVIDIA 610.43.02, Docker 29.6.1, NVIDIA Container Toolkit 1.19.1 |
| Model | `UD-Q3_K_XL-a5` (`scripts/get-model.sh`): 336.7 GB in 9 shards, from Unsloth's `UD-Q3_K_XL` at revision `346b3591` |

## How the numbers were taken

`scripts/bench.sh` against a freshly started server, one request at a time:

- **Decode:** one warmup, then 3 × 512-token answers each to a story prompt and a code prompt. Greedy, MTP drafting on, default reasoning effort (max): GLM-5.3 always thinks, so the answers are mostly reasoning tokens.
  The rate is taken over the decode phase only: tokens after the first, from the first streamed delta to the last.
- **Prompts:** one request per target size (4,096 / 28,672 / 57,344 tokens of filler text, which tokenize to 3.6K / 25K / 50K), each
  followed by 256 generated tokens. The prompt rate is the server's own `timings.prompt_per_second`, the median of 3 rounds.

__TABLES__
