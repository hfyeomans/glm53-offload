# glm53-offload: llama.cpp at a pinned upstream commit + this repo's patch (MoE expert offload for GLM-5.3 on one or two RTX PRO 6000).
#   docker build -t glm53-offload -f docker/Dockerfile .
# CUDA 13.0.0 on purpose (13.2 miscompiles IQ kernels on sm_120); sm_120 only (RTX PRO 6000 Blackwell).  The CPU side is built for
# AVX-512 (Zen 4/5 Threadripper / EPYC, Xeon W-3400+); on a CPU without it: --build-arg CPU_FLAGS="-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON".
ARG CUDA_DEVEL=nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d
ARG CUDA_RUNTIME=nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

FROM ${CUDA_DEVEL} AS build
ARG LLAMA_COMMIT=d02b4c0d66796e0a44380611d038123fefb9fe83
ARG CUDA_ARCHS=120
ARG CPU_FLAGS="-DGGML_AVX2=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_AVX512_VBMI=ON -DGGML_FMA=ON -DGGML_F16C=ON"
RUN apt-get update && apt-get install -y --no-install-recommends git cmake ninja-build build-essential libcurl4-openssl-dev ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp \
 && git fetch -q --depth 1 origin ${LLAMA_COMMIT} && git checkout -q FETCH_HEAD
COPY patches/glm53-offload.patch /tmp/glm53-offload.patch
RUN git apply --whitespace=nowarn /tmp/glm53-offload.patch
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" -DGGML_NATIVE=OFF ${CPU_FLAGS} \
      -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=OFF \
 && cmake --build build --config Release --target llama-server llama-quantize llama-perplexity

FROM ${CUDA_RUNTIME}
RUN apt-get update && apt-get install -y --no-install-recommends libcurl4 libgomp1 ca-certificates curl python3 util-linux \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/llama-server /src/build/bin/llama-quantize /src/build/bin/llama-perplexity /app/
COPY recipes/ /app/recipes/
COPY docker/serve.sh /app/serve.sh
RUN chmod +x /app/serve.sh
WORKDIR /app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s CMD curl -fs http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["/app/serve.sh"]
