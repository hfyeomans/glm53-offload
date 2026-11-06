#!/bin/bash
# get-model.sh -- fetch Unsloth's GLM-5.3 UD-Q3_K_XL at a pinned revision, verify it, and requantize it to the "a5" variant the recipes
# were measured on: the Q8_0 non-expert tensors (attention, shared experts, dense FFN: 19.5 GB read on EVERY token) to Q5_K with
# Unsloth's own importance matrix; every routed expert, the embedding and the output head are copied byte for byte.  Decode +11-14 %,
# perplexity +0.17 % (inside the error bar).  The requant takes ~2-5 min on a 64-core CPU, no GPU.
#
#   scripts/get-model.sh <dir> [--image glm53-offload:latest] [--delete-original] [--skip-download]
#
# Disk: ~343 GB (original) + 1.1 GB (imatrix) + ~337 GB (a5) = ~681 GB; --delete-original removes the first after a verified requant.
# Output: <dir>/UD-Q3_K_XL-a5/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf (pass that to run.sh --model).
# Downloads with the Hugging Face CLI (`hf`, from `pip install -U "huggingface_hub[hf_xet]"`) if it is on PATH, else in a python
# container.  A gated or rate-limited download needs HF_TOKEN in the environment.
set -u
DIR=; IMAGE=${GLM53_IMAGE:-glm53-offload:latest}; DEL=0; SKIP=0
while [ $# -gt 0 ]; do case $1 in
  --image) IMAGE=$2; shift 2;;
  --delete-original) DEL=1; shift;;
  --skip-download) SKIP=1; shift;;
  -h|--help) sed -n 2,13p "$0"; exit 0;;
  *) DIR=$1; shift;;
esac; done
[ -n "$DIR" ] || { echo "usage: $0 <dir> (see --help)"; exit 2; }
mkdir -p "$DIR" && DIR=$(cd "$DIR" && pwd)
REPO=unsloth/GLM-5.3-GGUF
REV=346b3591c7f28d1a23716f97a065ecf12ec14771
SRC=$DIR/UD-Q3_K_XL; DST=$DIR/UD-Q3_K_XL-a5; IMX=$DIR/imatrix_unsloth.gguf_file
IMX_SHA=936ec188a262f963e9b68bef0679758a0a80c5858e3a55a4596b45006696dec0
SHA=(   # the Hub's LFS sha256 of each shard at $REV
  56d6d59fc554a84c503c2f786e6978d55681e42b8c01370afd923a6286d05e0a
  4a66db8d7f441a948d435e7559a4080a58cc57a8cb35ce9104103ea4039cf34f
  03c626e0fe1cb5e9f245afa092df6c25fa1f55b1072ec65f143f804cec04b710
  d391c041f80c0fdf6cf92777528f642a1e6882562e5f0727cf35e00be7c8f686
  34f022b34bc4be26bec87e3ffb7b832a4a357f550c748c9a0013afbf0ce01918
  6961dbf52a18a6806ac3e399617fb3a6faeaca8d7264241f38cf972ef94bb5b7
  0b70151675f930eed91c83e51aa8c977657c4fe371af1a02951dd1b4e7258a68
  04eaa6d2cb5a548854c719994816a791a4e3d7783f38979e95b389f8f840badf
  6227252600ff83f185b6ad149266fbd0cb1c1f57ae0bfb71c247ee49faa3c31b )
A5_SHA=(   # the a5 shards this script produced on the reference machine (the requant is deterministic)
  0bec738e5b71267d7c68eb22440f957424daa56f8d35a3eedd59c370199d2bf4
  3804657b129d1e28f9c9ed952d03b294fa9d2633f22ad82407f1b5c43b745938
  78e1b33b309ee7d04ebb3a235b4f47ce6a984446bbc5f764fb7799bd54c98836
  6544fd57ec3e4f6957c1ec2905badcebc91a4309c63bfd7196c2fc956ac5fd65
  133092e64a5a043aab15e0221d1b9899ecf605b0f9d73d09ceee1d2f6f683dd3
  299021df69cda056d9dfbf9481b03df3694bccb82b3e04bf066d36aafe69fb19
  33bbea7d38949b27922e4c36249d7f56569f83cbb11b768ffa508b298cae69a6
  0dcc83781482a608f493e287e2b9288eb0c00ffb6ac896bc4f7d4d5745907d46
  d90504f4d42340214b98ace8b9cdc570d3ef4607cf0b3c66de5c95356c97fa69 )

say(){ echo "[get-model] $*"; }
avail_gb(){ df -BG --output=avail "$DIR" | tail -1 | tr -dc 0-9; }

if [ $SKIP = 0 ]; then
  [ "$(avail_gb)" -ge 690 ] || say "warning: $(avail_gb) GB free under $DIR; ~681 GB are needed (343 + 1 + 337)"
  INC=(--include "UD-Q3_K_XL/*" --include "imatrix_unsloth.gguf_file")
  if command -v hf >/dev/null; then
    hf download "$REPO" --revision "$REV" "${INC[@]}" --local-dir "$DIR" || exit 3
  else
    say "no hf CLI on PATH: downloading in a python container"
    docker run --rm -e HF_TOKEN="${HF_TOKEN:-}" -v "$DIR":/out python:3.12-slim sh -c \
      "pip install -q -U 'huggingface_hub[hf_xet]' && hf download $REPO --revision $REV --include 'UD-Q3_K_XL/*' --include imatrix_unsloth.gguf_file --local-dir /out" || exit 3
  fi
fi

say "verifying the original shards and the imatrix (sha256, ~343 GB)"
BAD=0; PIDS=()
for i in 1 2 3 4 5 6 7 8 9; do
  f=$SRC/GLM-5.3-UD-Q3_K_XL-0000$i-of-00009.gguf
  [ -f "$f" ] || { say "missing $f"; BAD=1; continue; }
  ( got=$(sha256sum "$f" | cut -d' ' -f1); [ "$got" = "${SHA[$((i-1))]}" ] || { echo "[get-model] MISMATCH $f: $got"; exit 1; } ) &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p" || BAD=1; done
[ "$(sha256sum "$IMX" 2>/dev/null | cut -d' ' -f1)" = "$IMX_SHA" ] || { say "imatrix missing or MISMATCH at $IMX"; BAD=1; }
[ $BAD = 0 ] || { say "verification failed: re-run the download (it resumes), then this script with --skip-download"; exit 4; }
say "original verified"

docker image inspect "$IMAGE" >/dev/null 2>&1 || { say "no image $IMAGE (docker build -t glm53-offload -f docker/Dockerfile .)"; exit 2; }
mkdir -p "$DST"
TT=( --tensor-type 'blk\.8\.ffn_(gate|up)_exps=iq4_xs'  --tensor-type 'blk\.8\.ffn_down_exps=q5_k'     # layer 8 is IQ4_XS / IQ4_XS / Q5_K
     --tensor-type 'blk\.7[5-7]\.ffn_down_exps=q6_k'                                                   # layers 75-77: down is Q6_K
     --tensor-type 'blk\.78\.ffn_(gate|up)_exps=q3_k'   --tensor-type 'blk\.78\.ffn_down_exps=q4_k'     # the NextN (MTP) layer
     --tensor-type 'ffn_(gate|up)_exps=iq3_xxs'         --tensor-type 'ffn_down_exps=iq4_xs'            # the other 71 MoE layers
     --tensor-type 'indexer=q8_0'                                                                       # the DSA indexer stays Q8_0 (before the attn_ rule)
     --tensor-type 'attn_=q5_k' --tensor-type 'shexp=q5_k' --tensor-type 'ffn_(gate|up|down)\.weight=q5_k' )
# the first matching --tensor-type wins and a tensor whose target equals its type is copied unchanged, so only the Q8_0 non-expert
# tensors change (attn_k_b, 192 columns, falls back to Q5_1)
say "requantizing to $DST"
docker run --rm -v "$DIR":/m --entrypoint /app/llama-quantize "$IMAGE" --allow-requantize --keep-split \
  --imatrix /m/imatrix_unsloth.gguf_file "${TT[@]}" --output-tensor-type q6_k --token-embedding-type q8_0 \
  /m/UD-Q3_K_XL/GLM-5.3-UD-Q3_K_XL-00001-of-00009.gguf /m/UD-Q3_K_XL-a5/GLM-5.3-UD-Q3_K_XL-a5 Q5_K_M "$(nproc)" || exit 5

say "verifying the a5 shards"
BAD=0
for i in 1 2 3 4 5 6 7 8 9; do
  f=$DST/GLM-5.3-UD-Q3_K_XL-a5-0000$i-of-00009.gguf
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "${A5_SHA[$((i-1))]}" ] || { say "a5 shard $i differs from the reference ($got): the model still runs, but it is not the measured file"; BAD=1; }
done
[ $BAD = 0 ] && say "a5 verified: byte-identical to the reference"
if [ $DEL = 1 ] && [ $BAD = 0 ]; then rm -rf "$SRC"; say "removed the original ($SRC)"; fi
say "done: $DST/GLM-5.3-UD-Q3_K_XL-a5-00001-of-00009.gguf"
