# Qwen3 Reranker inference snap

[![qwen3-reranker](https://snapcraft.io/qwen3-reranker/badge.svg)](https://snapcraft.io/qwen3-reranker)

<!-- README-I18N:START -->

**English** | [Português (Brasil)](./README.pt-BR.md) | [简体中文](./README.zh-CN.md)

<!-- README-I18N:END -->

**[Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)** is
[the Qwen team's](https://huggingface.co/Qwen) family of text reranking models (0.6B, 4B, 8B),
built for the second stage of retrieval pipelines: given a query and candidate documents, it
scores how relevant each document is. It supports 100+ languages (including programming
languages), 32K-token inputs and user-defined instructions, and shares the multilingual and
long-text strengths of the Qwen3 foundation models.

Use this snap to quickly install an optimized environment for local text reranking with
Qwen3-Reranker, served through a Jina/Cohere-compatible `/v1/rerank` API.

The snap includes the following hardware-optimized inference engines:

* cpu: Optimized for x64 and ARM (armv8, armv9) CPUs
* nvidia-gpu: CUDA-enabled GPU acceleration
* amd-gpu: ROCm-enabled GPU acceleration for AMD GPUs

The most suitable engine is automatically selected based on the available hardware.

#### Install
```shell
sudo snap install qwen3-reranker
```

#### Run
```shell
qwen3-reranker
```

> [!TIP]
> Some accelerators require extra [drivers](https://documentation.ubuntu.com/inference-snaps/how-to/setup/drivers/) to be usable with this snap.
> The `amd-gpu` engine uses the in-kernel `amdgpu` driver (KFD); the ROCm
> user-space libraries are bundled in the snap, so no host ROCm install is needed.

Rerank right away:

```shell
curl -s http://127.0.0.1:8366/v1/rerank -H 'Content-Type: application/json' -d '{
  "model": "qwen3-reranker-0-6b",
  "query": "What is the capital of China?",
  "documents": [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other."
  ],
  "top_n": 2}'
```

```json
{"results": [{"index": 0, "relevance_score": 0.9996},
             {"index": 1, "relevance_score": 0.0000046}]}
```

## Model sizes

Three sizes are available. 0.6B is the default; switch with
`sudo qwen3-reranker use-model <model>` and restart the service:

| Model | Weights | Disk (installed) | Highlights (official benchmarks) |
| --- | --- | --- | --- |
| `qwen3-reranker-0-6b` (Q8_0) | 0.64 GB | ~1.0 GB | MTEB-R 65.80, MMTEB-R 66.36 — beats open rerankers 10× its size |
| `qwen3-reranker-4b` (Q8_0) | 4.28 GB | ~4.7 GB | Best English MTEB-R 69.76 and FollowIR 14.84 |
| `qwen3-reranker-8b` (Q4_K_M) | 4.68 GB | ~5.1 GB | Best Chinese CMTEB-R 77.45, multilingual MMTEB-R 72.94, long-doc MLDR 70.19, code 81.22 |

```shell
sudo qwen3-reranker use-model qwen3-reranker-8b
sudo snap restart qwen3-reranker
```

## Requirements

Peak RSS of the `llama-server` process on the cpu engine, measured with context 8192 and the
default `ubatch-size` of 2048:

| Model | Idle | While scoring a ~1 700-token document |
| --- | --- | --- |
| 0.6B (Q8_0) | ~1.6 GB | ~3.7 GB |
| 4B (Q8_0) | ~5.3 GB | ~7.5 GB |
| 8B (Q4_K_M) | ~8.9 GB | ~11.1 GB |

The context size is not fixed: llama.cpp's `--fit` sizes it to the memory available, using
`--fit-ctx` (default 4096) as the floor, so tighter machines still serve smaller contexts.
Document scoring adds a transient ~2.2 GB at the default `ubatch-size`; raising `ubatch-size`
to 8192 raises that transient to ~9.6 GB (measured on 0.6B) — only do it with RAM to spare.

Indicative CPU throughput (12-core x86, other load present): ~1.9 short documents/s (0.6B),
~0.4 (4B), ~0.3 (8B). Reranking costs one full forward pass **per (query, document) pair** —
use it as the second stage over a bounded candidate set (typically 20–100), not over a whole
corpus.

Disk usage is ~1 GB once installed with the 0.6B model (snap + llama.cpp component + weights),
plus ~1.1 GB if the nvidia-gpu engine is selected (~730 MB download), or ~3.8 GB if the amd-gpu
engine is selected (~1.1 GB download). The 4B and 8B weights add ~4.3 GB / ~4.7 GB respectively.

## Usage guide

### Basic reranking

`POST /v1/rerank` (aliases: `/rerank`, `/reranking`, `/v1/reranking`, `/jinaai/rerank`) with
`query` (string), `documents` (string array) and optional `top_n`. Results come back sorted by
`relevance_score`, truncated to `top_n`. Sending `texts` instead of `documents` switches to the
TEI response format (`[{"index", "score"}]`, plus `"text"` with `return_text: true`).

### Score semantics

`relevance_score` is the softmaxed probability of the model answering "yes" to *"does this
document satisfy the query?"* — a number in [0, 1]. Scores are comparable **within one query**,
not calibrated across queries or models: pick cut-offs empirically per application, or just use
the ordering. Note that other stacks report different scales (sentence-transformers'
`CrossEncoder.predict` returns the raw logit difference, e.g. 7.625, unless you pass a Sigmoid
activation).

### Long documents and batches

* The maximum prompt (query + fixed template + document) must fit in **one physical batch**:
  with the default `ubatch-size` of 2048 that is roughly a 1 700–1 900-token document. Larger
  inputs are rejected with a clear error — never silently truncated. Truncate client-side, or
  raise the limit: `sudo qwen3-reranker set ubatch-size=8192 && sudo snap restart qwen3-reranker`
  (see the memory cost in Requirements). The model itself supports 32K-token pairs.
* Send all candidates of a query in **one request** (the server scores them as a batch);
  `top_n` trims the response.

### Two-stage RAG with local embeddings

Pair this snap with the sibling
[qwen3-embedding](https://snapcraft.io/qwen3-embedding) snap: embed your corpus once, retrieve
the top ~50 candidates per query with cosine similarity (port 8362), then rerank them here
(port 8366) and keep the top few. That is exactly the pipeline Qwen uses to evaluate the model.

## Configuration

The engine reads its settings from the snap configuration:

| Setting | Default | Purpose |
| --- | --- | --- |
| `min-context-size` | `4096` | Floor for `--fit-ctx`; `--fit` grows the context up to the model maximum |
| `sleep-idle-seconds` | `600` | Idle time before `llama-server` unloads the model |
| `ubatch-size` | `2048` | Maximum rerankable prompt size (also passed as `-b`; see Usage guide) |
| `http.port` / `http.host` | `8366` / `127.0.0.1` | Rerank API |

Read them with `qwen3-reranker get`; changing them needs root, and the service has to be
restarted to apply:

```shell
qwen3-reranker get
sudo qwen3-reranker set ubatch-size=8192
sudo snap restart qwen3-reranker
```

## Resources

📚 **[Documentation](https://documentation.ubuntu.com/inference-snaps/)**, learn how to use inference snaps

🤗 **[Model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)**, upstream documentation for the model

📦 **[GGUF weights](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)**, the converted weights this snap bundles

🧩 **[Qwen3-Embedding source](https://github.com/QwenLM/Qwen3-Embedding)**, upstream repository, benchmarks and deployment guides

📝 **[Qwen3 Embedding blog](https://qwenlm.github.io/blog/qwen3-embedding/)** and
**[paper (arXiv:2506.05176)](https://arxiv.org/abs/2506.05176)**

🐛 **[Issues](https://github.com/amenezes/qwen3reranker/issues)**, report bugs and request features

## Notes on this snap

* **Rerank-only server:** the engine runs `llama-server --rerank`. `/v1/rerank` is the interface;
  `/v1/embeddings` answers with degenerate (near-zero) vectors and `/v1/chat/completions` is not
  refused by the pinned runtime but replies with base-model nonsense — **use neither**.
* **No per-request instruction:** the reranker is instruction-aware (+1–5% typical gain), but
  llama.cpp applies the instruction baked into the GGUF's rerank template (the official default:
  *"Given a web search query, retrieve relevant passages that answer the query"*). The API has no
  instruct field; customizing it means rewriting the GGUF metadata or re-converting the model.
* **Weight source:** the Qwen team publishes no official GGUF for the rerankers, and **most
  community GGUFs are broken for reranking** (converted as plain causal LMs, they return
  meaningless scores — see
  [llama.cpp#16407](https://github.com/ggml-org/llama.cpp/issues/16407)). This snap pins the
  [Voodisss `-llama_cpp`](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)
  conversions (official llama.cpp converter, full reranker metadata), sha256-verified, with
  scores validated against the official model-card reference values. Do not swap GGUF sources
  without re-running `make rerank-test`.
* **Runtime pin:** llama.cpp is pinned to `b10344` — the family pin (minicpm5-2b,
  qwen3-embedding, qwen3guard). Qwen3-Reranker support requires builds newer than
  [PR #15824](https://github.com/ggml-org/llama.cpp/pull/15824) (2025-09-25). Bumps arrive as
  reviewed Renovate PRs (never automerged) and must pass the versioned E2E battery
  (`tests/rerank-e2e.py`).
* **Quantization:** Q8_0 for 0.6B/4B (near-lossless; −0.2/−0.3% NDCG per the converter's
  benchmark) and Q4_K_M for 8B (largest quant that fits the store's 5 GiB per-component limit;
  within noise of F16). Avoid anything below Q4_K_M for rerankers (Q2_K loses 4–29%).
* **Model license:** Apache-2.0 (Qwen team, Alibaba Group). The GGUF conversion (Voodisss) is
  Apache-2.0. The packaging code is GPL-3.0.
* **Engine selection:** the `nvidia-gpu` and `amd-gpu` engines are picked whenever a GPU
  from the matching vendor is present on the PCI bus, even if the driver is not usable.
  The backend then fails to initialise and llama.cpp falls back to the CPU backend; run
  `sudo qwen3-reranker use-engine cpu` to switch back and free the engine component
  (~1.1 GB NVIDIA, ~3.8 GB AMD). The AMD engine only matches the microarchitectures
  supported by the bundled ROCm build (gfx1010-gfx1201, including the RX 9070 XT);
  older AMD GPUs select the cpu engine. Selection runs at install time and needs the
  `hardware-observe` interface; if its auto-connection has not been granted, connect it
  and re-run the selection manually:
  ```shell
  sudo snap connect qwen3-reranker:hardware-observe
  sudo qwen3-reranker use-engine --auto
  ```
* **AMD engine needs `process-control`:** the ROCm runtime pins its worker threads with
  `sched_setaffinity`, and snapd's default seccomp policy only allows that call for pid 0.
  Without the plug, HSA fails to initialise, llama.cpp reports `no ROCm-capable device is
  detected` and runs on the CPU while the engine still shows `amd-gpu`. The interface is not
  auto-connected, so connect it once after install -- and request its auto-connection before
  publishing to the store:
  ```shell
  sudo snap connect qwen3-reranker:process-control
  sudo snap restart qwen3-reranker.server
  ```
* **Provider:** the snap exports an `inference-provider` content slot, so another snap can
  consume this endpoint instead of running its own reranker.
* **Local API:** the endpoint has no authentication and allows cross-origin requests. It listens
  on `127.0.0.1` only — do not rebind `http.host` to a public interface.

## Build and install from source

Clone the repo:
```shell
git clone https://github.com/amenezes/qwen3reranker
cd qwen3reranker
```

Initialize the development environment:
```shell
make init        # default 0.6B weights only (~0.64 GB)
make init-all    # all three sizes (~9.6 GB) — required before `make build`
```

`snapcraft` builds inside a container by default, so a build provider must be available first.
With LXD, initialize it once (`--storage-backend=dir` avoids a ZFS loop pool):

```shell
lxd init --auto --storage-backend=dir
```

Without a provider, build on the host with `sudo snapcraft pack --destructive-mode
--platform amd64` instead.

Build and install snap:
```shell
make build
make install
```

`make install` installs the snap and every `.comp` in the directory, reconnects the interfaces
and preserves the engine already selected. To change the engine on a fresh installation — or
after `./dev/install.sh --clean`, which wipes the configuration — pass `--engine`:

```shell
./dev/install.sh --engine=amd-gpu
```

Test the installed snap:
```shell
make smoke-test                                  # framework basics (engine, config, port)
make rerank-test                                 # 16-check E2E battery on the default model
make rerank-test MODEL=qwen3-reranker-8b         # after `use-model qwen3-reranker-8b`
```

To use a local copy of the weights, place the files under `components/` as documented in each
component's `README.md` and run `make verify-model` / `make verify-extra-models`.
