# Qwen3-Reranker-4B.Q8_0.gguf

Model weights for [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B), the 4B
multilingual text reranker from the Qwen team (Alibaba Group), converted to GGUF with the
official llama.cpp converter and published in
[Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp](https://huggingface.co/Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp).

Q8_0 is near-lossless (community NDCG@10 benchmark on the model card: −0.3% vs F16) and at
4.28 GB it still fits the Snap Store's 5 GiB per-component limit with ~20% margin, so no
sharding or lower quant is needed for this size.

> [!IMPORTANT]
> Do **not** substitute a GGUF from another community repo without re-verifying it: most were
> converted as plain causal LMs and lack the reranker metadata (`cls.output.weight`, RANK
> pooling, named `rerank` template), producing meaningless scores with `llama-server --rerank`
> ([llama.cpp#16407](https://github.com/ggml-org/llama.cpp/issues/16407)). See the 0.6B
> component README for the full list of required metadata keys.

The weights are **not** committed (`.gitignore` excludes `*.gguf`). `make init-all` downloads
them into this directory and verifies the pinned checksum:

```shell
make download-extra-models   # hf download $(GGUF_4B_REPO) $(GGUF_4B_FILE) --local-dir components/model-4b-q8-0-gguf
make verify-extra-models     # sha256sum --check against GGUF_4B_SHA256 in the Makefile
```

| Field | Value |
| --- | --- |
| File | `Qwen3-Reranker-4B.Q8_0.gguf` |
| Size | 4 279 678 912 bytes (4081 MiB) |
| sha256 | `102ac400d68f02877c2fdfecf0732872652298448352b6faf28ec7f8dca8c913` |
| Quantization | Q8_0 (single file, no shards) |
| GGUF metadata | `general.architecture=qwen3`, `qwen3.pooling_type=4` (RANK), `qwen3.classifier.output_labels=["yes","no"]`, named template `tokenizer.chat_template.rerank`, ctx 40960 |
| Rerank head | `cls.output.weight` [2560, 2] (yes/no rows of the tied lm_head) |
| License | Apache-2.0 (Qwen team, Alibaba Group; conversion: Voodisss) |

Single-file GGUF, so `models/qwen3-reranker-4b/model.yaml` needs no `layout`/symlinks: the weight
is mounted by the `model-4b-q8-0-gguf` component at `$SNAP_COMPONENTS/model-4b-q8-0-gguf/` and
exposed as `MODEL_FILE`.

## Why this README is tracked

`snap/snapcraft.yaml` has a `local-component-files` part with `source: components`. Keeping this
directory in git makes a fresh clone buildable while the 4.3 GB weight stays out of the
repository. The part organizes only the `.gguf` into the component and prunes everything else, so
this file is never shipped inside the snap.
