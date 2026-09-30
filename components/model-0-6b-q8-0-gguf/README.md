# Qwen3-Reranker-0.6B.Q8_0.gguf

Model weights for [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), the
0.6B multilingual text reranker from the Qwen team (Alibaba Group), converted to GGUF with the
official llama.cpp converter and published in
[Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp).

Q8_0 is near-lossless at this scale (community NDCG@10 benchmark on the model card: −0.2% vs
F16) and small (~0.64 GB), which is why it is the default variant of this snap.

> [!IMPORTANT]
> Do **not** substitute a GGUF from another community repo without re-verifying it. Most
> Qwen3-Reranker GGUFs (mradermacher, QuantFactory, Mungert, DevQuasar, …) were converted as
> plain causal LMs: they lack the `cls.output.weight` yes/no head, the RANK pooling metadata and
> the named `rerank` chat template, so `llama-server --rerank` returns meaningless scores with
> them ([llama.cpp#16407](https://github.com/ggml-org/llama.cpp/issues/16407)). The required
> metadata is listed in the table below and checked by `make verify-model` (checksum) plus the
> E2E battery (`make rerank-test`, reference scores from the official model card).

The weights are **not** committed (`.gitignore` excludes `*.gguf`). `make init` downloads them
into this directory and verifies the pinned checksum:

```shell
make download-models   # hf download $(GGUF_06B_REPO) $(GGUF_06B_FILE) --local-dir components/model-0-6b-q8-0-gguf
make verify-model      # sha256sum --check against GGUF_06B_SHA256 in the Makefile
```

To build offline (or from an internal mirror), place the file here yourself and run
`make verify-model`; the `GGUF_06B_REPO`/`GGUF_06B_FILE`/`GGUF_06B_SHA256` variables in the
`Makefile` are overridable.

| Field | Value |
| --- | --- |
| File | `Qwen3-Reranker-0.6B.Q8_0.gguf` |
| Size | 639 153 344 bytes (610 MiB) |
| sha256 | `6ddab39a36c6c87fdb76f0e5f05657012d5dbc97034c0983c157f17ef9f34d55` |
| Quantization | Q8_0 (single file, no shards) |
| GGUF metadata | `general.architecture=qwen3`, `qwen3.pooling_type=4` (RANK), `qwen3.classifier.output_labels=["yes","no"]`, named template `tokenizer.chat_template.rerank`, ctx 40960 |
| Rerank head | `cls.output.weight` [1024, 2] (yes/no rows of the tied lm_head) |
| License | Apache-2.0 (Qwen team, Alibaba Group; conversion: Voodisss) |

Single-file GGUF, so `models/qwen3-reranker-0-6b/model.yaml` needs no `layout`/symlinks: the
weight is mounted by the `model-0-6b-q8-0-gguf` component at
`$SNAP_COMPONENTS/model-0-6b-q8-0-gguf/` and exposed as `MODEL_FILE`.

## Why this README is tracked

`snap/snapcraft.yaml` has a `local-component-files` part with `source: components`. Keeping this
directory in git makes a fresh clone buildable (and fails with a clear "file not found" instead of
a missing source directory) while the 0.64 GB weight stays out of the repository. The part
organizes only the `.gguf` into the component and prunes everything else, so this file is never
shipped inside the snap.
