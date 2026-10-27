# Qwen3 Reranker 推理 snap

[![qwen3-reranker](https://snapcraft.io/qwen3-reranker/badge.svg)](https://snapcraft.io/qwen3-reranker)

<!-- README-I18N:START -->

[English](./README.md) | [Português (Brasil)](./README.pt-BR.md) | **简体中文**

<!-- README-I18N:END -->

**[Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)** 是
[通义千问团队](https://huggingface.co/Qwen)推出的文本重排序模型系列（0.6B、4B、8B），
专为检索流水线的第二阶段构建：给定一个查询与候选文档，为每个文档的相关性打分。
支持 100 多种语言（含编程语言）、32K token 输入与用户自定义指令，并共享 Qwen3
基座模型在多语言与长文本方面的优势。

使用本 snap 可快速安装一个优化的本地文本重排序环境，通过 Jina/Cohere 兼容的
`/v1/rerank` API 提供服务。

本 snap 包含以下针对硬件优化的推理引擎：

* cpu：针对 x64 与 ARM（armv8、armv9）CPU 优化
* nvidia-gpu：CUDA GPU 加速
* amd-gpu：AMD GPU 的 ROCm 加速

系统会根据可用硬件自动选择最合适的引擎。

#### 安装
```shell
sudo snap install qwen3-reranker
```

#### 运行
```shell
qwen3-reranker
```

> [!TIP]
> 部分加速器需要额外的[驱动](https://documentation.ubuntu.com/inference-snaps/how-to/setup/drivers/)才能与本 snap 配合使用。
> `amd-gpu` 引擎使用内核自带的 `amdgpu` 驱动（KFD）；ROCm 用户态库已打包在本
> snap 中，主机无需安装 ROCm。

立即重排序：

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

## 模型规格

提供三种规格。默认 0.6B；使用 `sudo qwen3-reranker use-model <模型>` 切换并重启服务：

| 模型 | 权重 | 安装后磁盘 | 亮点（官方基准测试） |
| --- | --- | --- | --- |
| `qwen3-reranker-0-6b` (Q8_0) | 0.64 GB | ~1.0 GB | MTEB-R 65.80、MMTEB-R 66.36——胜过规模为其 10 倍的开源重排序模型 |
| `qwen3-reranker-4b` (Q8_0) | 4.28 GB | ~4.7 GB | 英文最佳：MTEB-R 69.76 与 FollowIR 14.84 |
| `qwen3-reranker-8b` (Q4_K_M) | 4.68 GB | ~5.1 GB | 中文最佳：CMTEB-R 77.45；多语言 MMTEB-R 72.94、长文档 MLDR 70.19、代码 81.22 |

```shell
sudo qwen3-reranker use-model qwen3-reranker-8b
sudo snap restart qwen3-reranker
```

## 系统要求

cpu 引擎上 `llama-server` 进程的峰值 RSS，实测条件为上下文 8192 与默认
`ubatch-size` 2048：

| 模型 | 空闲 | 为约 1 700 token 的文档打分时 |
| --- | --- | --- |
| 0.6B (Q8_0) | ~1.6 GB | ~3.7 GB |
| 4B (Q8_0) | ~5.3 GB | ~7.5 GB |
| 8B (Q4_K_M) | ~8.9 GB | ~11.1 GB |

上下文大小并非固定：llama.cpp 的 `--fit` 会按可用内存调整上下文大小，以 `--fit-ctx`
（默认 4096）为下限，因此内存紧张的机器仍能以更小的上下文提供服务。在默认
`ubatch-size` 下，文档打分会带来约 2.2 GB 的瞬时增量；将 `ubatch-size` 提高到 8192
会使该瞬时至约 9.6 GB（0.6B 实测）——仅在内存充裕时这么做。

参考 CPU 吞吐量（12 核 x86，存在其他负载）：约 1.9 短文档/秒（0.6B）、约 0.4（4B）、
约 0.3（8B）。重排序对**每个（查询，文档）对**都需一次完整前向传播——请将其用作有界
候选集（通常 20–100 个）上的第二阶段，而不是对整个语料库使用。

安装 0.6B 模型后磁盘占用约 1 GB（snap + llama.cpp 组件 + 权重）；若选择 nvidia-gpu
引擎再加约 1.1 GB（约 730 MB 下载）；若选择 amd-gpu 引擎，再加约 3.8 GB（约 1.1 GB
下载）。4B 与 8B 权重分别再增加约 4.3 GB / 4.7 GB。

## 使用指南

### 基础重排序

`POST /v1/rerank`（别名：`/rerank`、`/reranking`、`/v1/reranking`、`/jinaai/rerank`），
请求参数为 `query`（字符串）、`documents`（字符串数组）与可选的 `top_n`。返回结果按
`relevance_score` 排序，并截断至 `top_n`。发送 `texts` 而非 `documents` 会切换到
TEI 响应格式（`[{"index", "score"}]`，当 `return_text: true` 时附带 `"text"`）。

### 分数语义

`relevance_score` 是模型对*"该文档是否满足查询？"*回答"是"的 softmax 概率——取值位于
[0, 1] 区间。分数仅在**同一查询内**可比，跨查询或跨模型并未校准：请按应用通过实验确定
截断阈值，或干脆只使用排序结果。注意其他技术栈的取值范围不同（sentence-transformers 的
`CrossEncoder.predict` 返回原始 logit 差值，例如 7.625，除非传入 Sigmoid 激活函数）。

### 长文档与批量

* 最大提示（查询 + 固定模板 + 文档）必须容纳在**一个物理批次**内：默认 `ubatch-size`
  2048 下约相当于 1 700–1 900 token 的文档。更大的输入会被明确报错拒绝——绝不会被
  静默截断。请在客户端截断，或提高上限：
  `sudo qwen3-reranker set ubatch-size=8192 && sudo snap restart qwen3-reranker`
  （内存代价见系统要求）。模型本身支持 32K token 的输入对。
* 将一个查询的所有候选放入**一次请求**发送（服务端按批打分）；`top_n` 用于裁剪响应。

### 结合本地嵌入的两阶段 RAG

将本 snap 与兄弟 snap [qwen3-embedding](https://snapcraft.io/qwen3-embedding) 搭配：
对语料库做一次嵌入，用余弦相似度为每个查询检索约前 50 个候选（端口 8362），再在此处
（端口 8366）重排序并保留最靠前的几个。这正是通义千问团队评测该模型所用的流水线。

## 配置

引擎从 snap 配置读取设置：

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `min-context-size` | `4096` | `--fit-ctx` 下限；`--fit` 会增长上下文至模型上限 |
| `sleep-idle-seconds` | `600` | 空闲多久后 `llama-server` 卸载模型 |
| `ubatch-size` | `2048` | 可重排序的最大提示长度（同时作为 `-b` 传递；见使用指南） |
| `http.port` / `http.host` | `8366` / `127.0.0.1` | 重排序 API |

使用 `qwen3-reranker get` 查看；修改需要 root，且需重启服务生效：

```shell
qwen3-reranker get
sudo qwen3-reranker set ubatch-size=8192
sudo snap restart qwen3-reranker
```

## 资源

📚 **[文档](https://documentation.ubuntu.com/inference-snaps/)**，学习如何使用 inference snaps

🤗 **[模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)**，模型官方文档

📦 **[GGUF 权重](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)**，本 snap 内置的转换后权重

🧩 **[Qwen3-Embedding 仓库](https://github.com/QwenLM/Qwen3-Embedding)**，上游仓库、基准测试与部署指南

📝 **[Qwen3 Embedding 博客](https://qwenlm.github.io/blog/qwen3-embedding/)** 与
**[论文（arXiv:2506.05176）](https://arxiv.org/abs/2506.05176)**

🐛 **[Issues](https://github.com/amenezes/qwen3reranker/issues)**，报告问题与需求

## 关于本 snap 的说明

* **仅重排序服务端：** 引擎以 `llama-server --rerank` 运行。`/v1/rerank` 是唯一接口；
  `/v1/embeddings` 会返回退化（近零）向量，`/v1/chat/completions` 不会被固定版本的
  运行时拒绝，但只会回复基座模型的无意义输出——**两者都请勿使用**。
* **不支持按请求指定指令：** 重排序模型具备指令感知能力（典型提升 1–5%），但 llama.cpp
  使用的是内置在 GGUF 重排序模板中的指令（官方默认值：*"Given a web search query,
  retrieve relevant passages that answer the query"*）。API 没有 instruct 字段；
  自定义指令意味着重写 GGUF 元数据或重新转换模型。
* **权重来源：** 通义千问团队未为重排序模型发布官方 GGUF，且**大多数社区 GGUF 无法
  正确重排序**（按普通因果语言模型转换，会返回无意义的分数——见
  [llama.cpp#16407](https://github.com/ggml-org/llama.cpp/issues/16407)）。本 snap
  固定使用 [Voodisss `-llama_cpp`](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)
  转换版本（官方 llama.cpp 转换器、完整重排序元数据），经 sha256 校验，分数已对照
  官方模型卡参考值验证。除非重新运行 `make rerank-test`，否则请勿更换 GGUF 来源。
* **运行时固定版本：** llama.cpp 固定为 `b10344`——家族统一固定版本（minicpm5-2b、
  qwen3-embedding、qwen3guard）。Qwen3-Reranker 支持要求晚于
  [PR #15824](https://github.com/ggml-org/llama.cpp/pull/15824)（2025-09-25）的构建。
  版本升级以经评审的 Renovate PR 形式提交（从不自动合并），且必须通过版本化 E2E
  测试套件（`tests/rerank-e2e.py`）。
* **量化：** 0.6B/4B 采用 Q8_0（近无损；按转换器基准测试 NDCG 变化 −0.2/−0.3%），
  8B 采用 Q4_K_M（符合 Snap Store 单组件 5 GiB 上限的最大量化档位；与 F16 的差距在
  噪声范围内）。重排序模型请避免低于 Q4_K_M 的量化（Q2_K 损失 4–29%）。
* **模型许可证：** Apache-2.0（通义千问团队，阿里巴巴集团）。GGUF 转换（Voodisss）为
  Apache-2.0。打包代码为 GPL-3.0。
* **引擎选择：** 只要 PCI 总线上存在对应厂商的 GPU，就会选择 `nvidia-gpu` 或
  `amd-gpu` 引擎，即使驱动不可用。此时后端初始化失败，llama.cpp 回退到 CPU 后端；运行
  `sudo qwen3-reranker use-engine cpu` 可切回并释放该引擎组件（NVIDIA 约 1.1 GB，
  AMD 约 3.8 GB）。AMD 引擎仅匹配内置 ROCm 构建支持的微架构
  （gfx1010–gfx1201，含 RX 9070 XT）；较旧的 AMD GPU 会选择 cpu 引擎。选择在安装时进行，
  需要 `hardware-observe` 接口；若未获自动连接，请手动连接后重新选择：
  ```shell
  sudo snap connect qwen3-reranker:hardware-observe
  sudo qwen3-reranker use-engine --auto
  ```
* **Provider：** 本 snap 导出 `inference-provider` 内容插槽，其他 snap 可直接消费该
  端点而无需自行运行重排序模型。
* **本地 API：** 端点无鉴权并允许跨域请求。仅监听 `127.0.0.1`——请勿将 `http.host`
  重绑定到公网接口。

## 从源码构建与安装

克隆仓库：
```shell
git clone https://github.com/amenezes/qwen3reranker
cd qwen3reranker
```

初始化开发环境：
```shell
make init        # default 0.6B weights only (~0.64 GB)
make init-all    # all three sizes (~9.6 GB) — required before `make build`
```

`snapcraft` 默认在容器内构建，因此需要先有构建 provider。使用 LXD 时初始化一次
（`--storage-backend=dir` 可避免 ZFS loop pool）：

```shell
lxd init --auto --storage-backend=dir
```

没有 provider 时，可在主机上用 `sudo snapcraft pack --destructive-mode --platform amd64`
构建。

构建并安装 snap：
```shell
make build
make install
```

测试已安装的 snap：
```shell
make smoke-test                                  # framework basics (engine, config, port)
make rerank-test                                 # 16-check E2E battery on the default model
make rerank-test MODEL=qwen3-reranker-8b         # after `use-model qwen3-reranker-8b`
```

如需使用本地权重副本，请按各组件 `README.md` 中的说明将文件放置到 `components/` 下，
并运行 `make verify-model` / `make verify-extra-models`。

## 发布到 Snap Store

有两个商店侧申请是前置条件，且都需要时间，请尽早提交：

* **组件上传权限** —— 权重与 llama.cpp 运行时以
  [components](https://snapcraft.io/docs/explanation/how-snaps-work/snap-components/)
  形式分发，组件上传仅限商店白名单账户。请在
  [Snapcraft 论坛](https://forum.snapcraft.io/c/store-requests/)申请。
* **`hardware-observe` 自动连接** —— 该接口默认不自动连接，因此在申请获批前，商店
  安装时 `use-engine --auto` 无法检测硬件（参见
  [流程](https://snapcraft.io/docs/reference/administration/process-for-aliases-auto-connections-and-tracks/)）。

首次上传还需要已注册名称与真实存在的仓库：商店页面会链接 `snap/snapcraft.yaml` 中的
`website`、`source-code`、`issues` 与 `contact`。

```shell
snapcraft login
snapcraft register qwen3-reranker
make build
./dev/upload.sh latest/edge   # snapcraft upload <snap> --component ... --release=latest/edge
```

配置好 `SNAPCRAFT_STORE_CREDENTIALS` secret（`snapcraft export-login` 的输出）后，
`.github/workflows/cicd.yaml` 会在每次 push 到 `main` 时发布到 `latest/edge`。
