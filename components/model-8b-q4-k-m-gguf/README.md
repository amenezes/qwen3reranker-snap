# Qwen3-Reranker-8B-Q4_K_M.gguf

Model weights for [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B), the 8B
multilingual text reranker from the Qwen team (Alibaba Group), converted to GGUF with the
official llama.cpp converter and published in
[Voodisss/Qwen3-Reranker-8B-GGUF-llama_cpp](https://huggingface.co/Voodisss/Qwen3-Reranker-8B-GGUF-llama_cpp).

**Why Q4_K_M:** the Q8_0 build (8.05 GB), Q6_K (6.21 GB), Q5_K_M (5.42 GB) and even Q5_0
(5.29 GB) exceed or crowd the Snap Store's 5 GiB per-component limit; Q4_K_M (4.68 GB, ~13%
margin) is the largest quantization that ships as a single component, and the community NDCG@10
benchmark puts it within noise of F16 (+0.8% on the 4B sibling). Q8_0 with 2-way sharding (like
`glm-4.7-flash-snap`) is recorded as a possible evolution in `ai/07-decisoes.md` (DEC-03).

> [!IMPORTANT]
> Do **not** substitute a GGUF from another community repo without re-verifying it: most were
> converted as plain causal LMs and lack the reranker metadata (`cls.output.weight`, RANK
> pooling, named `rerank` template), producing meaningless scores with `llama-server --rerank`
> ([llama.cpp#16407](https://github.com/ggml-org/llama.cpp/issues/16407)). See the 0.6B
> component README for the full list of required metadata keys.

The weights are **not** committed (`.gitignore` excludes `*.gguf`). `make init-all` downloads
them into this directory and verifies the pinned checksum:

```shell
make download-extra-models   # hf download $(GGUF_8B_REPO) $(GGUF_8B_FILE) --local-dir components/model-8b-q4-k-m-gguf
make verify-extra-models     # sha256sum --check against GGUF_8B_SHA256 in the Makefile
```

| Field | Value |
| --- | --- |
| File | `Qwen3-Reranker-8B-Q4_K_M.gguf` |
| Size | 4 676 825 344 bytes (4460 MiB) |
| sha256 | `b0de7188a5801565d87c895d4ee8b6a1da3f46de3f07f378b831097894f210d2` |
| Quantization | Q4_K_M (single file, no shards) |
| GGUF metadata | `general.architecture=qwen3`, `qwen3.pooling_type=4` (RANK), `qwen3.classifier.output_labels=["yes","no"]`, named template `tokenizer.chat_template.rerank`, ctx 40960 |
| Rerank head | `cls.output.weight` [4096, 2] (yes/no rows of lm_head) |
| License | Apache-2.0 (Qwen team, Alibaba Group; conversion: Voodisss) |

Single-file GGUF, so `models/qwen3-reranker-8b/model.yaml` needs no `layout`/symlinks: the weight
is mounted by the `model-8b-q4-k-m-gguf` component at `$SNAP_COMPONENTS/model-8b-q4-k-m-gguf/`
and exposed as `MODEL_FILE`.

## Why this README is tracked

`snap/snapcraft.yaml` has a `local-component-files` part with `source: components`. Keeping this
directory in git makes a fresh clone buildable while the 4.7 GB weight stays out of the
repository. The part organizes only the `.gguf` into the component and prunes everything else, so
this file is never shipped inside the snap.
