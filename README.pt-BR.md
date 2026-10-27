# Snap de inferência Qwen3 Reranker

[![qwen3-reranker](https://snapcraft.io/qwen3-reranker/badge.svg)](https://snapcraft.io/qwen3-reranker)

<!-- README-I18N:START -->

[English](./README.md) | **Português (Brasil)** | [简体中文](./README.zh-CN.md)

<!-- README-I18N:END -->

**[Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)** é a família de modelos de
*reranking* de texto da [equipe Qwen](https://huggingface.co/Qwen) (0.6B, 4B, 8B), projetada para
o segundo estágio de pipelines de recuperação: dada uma consulta e documentos candidatos,
pontua quão relevante é cada documento. Suporta mais de 100 idiomas (incluindo linguagens de
programação), entradas de 32K tokens e instruções definidas pelo usuário, e compartilha os
pontos fortes multilíngues e de textos longos dos modelos base Qwen3.

Use este snap para instalar rapidamente um ambiente otimizado para *reranking* de texto local
com Qwen3-Reranker, servido por uma API compatível com Jina/Cohere em `/v1/rerank`.

O snap inclui os seguintes motores de inferência otimizados por hardware:

* cpu: Otimizado para CPUs x64 e ARM (armv8, armv9)
* nvidia-gpu: Aceleração de GPU via CUDA
* amd-gpu: aceleração por GPU com ROCm para GPUs AMD

O motor mais adequado é selecionado automaticamente com base no hardware disponível.

#### Instalação
```shell
sudo snap install qwen3-reranker
```

#### Execução
```shell
qwen3-reranker
```

> [!TIP]
> Alguns aceleradores exigem [drivers](https://documentation.ubuntu.com/inference-snaps/how-to/setup/drivers/) adicionais para funcionar com este snap.
> O motor `amd-gpu` usa o driver `amdgpu` do kernel (KFD); as bibliotecas de user
> space do ROCm vêm embarcadas no snap, então não é preciso instalar ROCm no host.

Faça *reranking* imediatamente:

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

## Tamanhos de modelo

Três tamanhos estão disponíveis. O 0.6B é o padrão; troque com
`sudo qwen3-reranker use-model <modelo>` e reinicie o serviço:

| Modelo | Pesos | Disco (instalado) | Destaques (benchmarks oficiais) |
| --- | --- | --- | --- |
| `qwen3-reranker-0-6b` (Q8_0) | 0,64 GB | ~1,0 GB | MTEB-R 65.80, MMTEB-R 66.36 — supera *rerankers* abertos 10× maiores |
| `qwen3-reranker-4b` (Q8_0) | 4,28 GB | ~4,7 GB | Melhor em inglês: MTEB-R 69.76 e FollowIR 14.84 |
| `qwen3-reranker-8b` (Q4_K_M) | 4,68 GB | ~5,1 GB | Melhor em chinês: CMTEB-R 77.45, multilíngue MMTEB-R 72.94, documentos longos MLDR 70.19, código 81.22 |

```shell
sudo qwen3-reranker use-model qwen3-reranker-8b
sudo snap restart qwen3-reranker
```

## Requisitos

Pico de RSS do processo `llama-server` no motor cpu, medido com contexto 8192 e o
`ubatch-size` padrão de 2048:

| Modelo | Ocioso | Ao pontuar um documento de ~1.700 tokens |
| --- | --- | --- |
| 0.6B (Q8_0) | ~1,6 GB | ~3,7 GB |
| 4B (Q8_0) | ~5,3 GB | ~7,5 GB |
| 8B (Q4_K_M) | ~8,9 GB | ~11,1 GB |

O tamanho do contexto não é fixo: o `--fit` do llama.cpp o dimensiona conforme a memória
disponível, usando o `--fit-ctx` (padrão 4096) como piso, então máquinas mais limitadas ainda
servem contextos menores. A pontuação de documentos adiciona ~2,2 GB transitórios com o
`ubatch-size` padrão; elevar o `ubatch-size` para 8192 eleva esse transitório para ~9,6 GB
(medido no 0.6B) — só faça isso se houver RAM de sobra.

Throughput indicativo em CPU (x86 de 12 núcleos, com outras cargas presentes): ~1,9 documentos
curtos/s (0.6B), ~0,4 (4B), ~0,3 (8B). O *reranking* custa um *forward pass* completo **por par
(consulta, documento)** — use-o como segundo estágio sobre um conjunto limitado de candidatos
(tipicamente 20–100), não sobre um corpus inteiro.

O uso de disco é de ~1 GB após a instalação com o modelo 0.6B (snap + componente llama.cpp +
pesos), mais ~1,1 GB se o motor nvidia-gpu for selecionado (~730 MB de download), ou ~3,8 GB se o
motor amd-gpu for selecionado (~1,1 GB de download). Os pesos 4B e 8B adicionam ~4,3 GB / ~4,7 GB,
respectivamente.

## Guia de uso

### Reranking básico

`POST /v1/rerank` (aliases: `/rerank`, `/reranking`, `/v1/reranking`, `/jinaai/rerank`) com
`query` (string), `documents` (array de strings) e `top_n` opcional. Os resultados voltam
ordenados por `relevance_score`, truncados para `top_n`. Enviar `texts` em vez de `documents`
alterna para o formato de resposta do TEI (`[{"index", "score"}]`, mais `"text"` com
`return_text: true`).

### Semântica dos scores

`relevance_score` é a probabilidade com softmax de o modelo responder "sim" à pergunta *"este
documento satisfaz a consulta?"* — um número em [0, 1]. Os scores são comparáveis **dentro de
uma mesma consulta**, não calibrados entre consultas ou modelos: escolha pontos de corte
empiricamente por aplicação, ou apenas use a ordenação. Note que outras stacks reportam escalas
diferentes (o `CrossEncoder.predict` do sentence-transformers retorna a diferença bruta de
logits, ex. 7.625, a menos que você passe uma ativação Sigmoid).

### Documentos longos e lotes

* O prompt máximo (consulta + template fixo + documento) deve caber em **um lote físico**: com o
  `ubatch-size` padrão de 2048, isso equivale a aproximadamente um documento de 1.700–1.900
  tokens. Entradas maiores são rejeitadas com um erro claro — nunca truncadas silenciosamente.
  Trunque no cliente, ou eleve o limite:
  `sudo qwen3-reranker set ubatch-size=8192 && sudo snap restart qwen3-reranker` (veja o custo
  de memória em Requisitos). O modelo em si suporta pares de 32K tokens.
* Envie todos os candidatos de uma consulta em **uma requisição** (o servidor os pontua como um
  lote); o `top_n` trunca a resposta.

### RAG em dois estágios com embeddings locais

Combine este snap com o snap irmão [qwen3-embedding](https://snapcraft.io/qwen3-embedding): faça
o *embedding* do seu corpus uma vez, recupere os ~50 melhores candidatos por consulta com
similaridade cosseno (porta 8362), depois faça o *reranking* deles aqui (porta 8366) e mantenha
os primeiros. Esse é exatamente o pipeline que a Qwen usa para avaliar o modelo.

## Configuração

O motor lê suas configurações da configuração do snap:

| Configuração | Padrão | Propósito |
| --- | --- | --- |
| `min-context-size` | `4096` | Piso do `--fit-ctx`; o `--fit` cresce o contexto até o máximo do modelo |
| `sleep-idle-seconds` | `600` | Tempo ocioso antes de o `llama-server` descarregar o modelo |
| `ubatch-size` | `2048` | Tamanho máximo de prompt para *reranking* (também passado como `-b`; veja o Guia de uso) |
| `http.port` / `http.host` | `8366` / `127.0.0.1` | API de rerank |

Leia com `qwen3-reranker get`; alterá-las exige root e o serviço precisa ser reiniciado para
aplicar:

```shell
qwen3-reranker get
sudo qwen3-reranker set ubatch-size=8192
sudo snap restart qwen3-reranker
```

## Recursos

📚 **[Documentação](https://documentation.ubuntu.com/inference-snaps/)**, aprenda a usar inference snaps

🤗 **[Model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)**, documentação oficial do modelo

📦 **[Pesos GGUF](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)**, os pesos convertidos que este snap embute

🧩 **[Repositório Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)**, upstream com benchmarks e guias de implantação

📝 **[Blog do Qwen3 Embedding](https://qwenlm.github.io/blog/qwen3-embedding/)** e
**[artigo (arXiv:2506.05176)](https://arxiv.org/abs/2506.05176)**

🐛 **[Issues](https://github.com/amenezes/qwen3reranker/issues)**, reporte bugs e peça recursos

## Notas sobre este snap

* **Servidor exclusivo de rerank:** o motor roda `llama-server --rerank`. A interface é
  `/v1/rerank`; `/v1/embeddings` responde com vetores degenerados (próximos de zero) e
  `/v1/chat/completions` não é recusado pelo runtime fixado, mas responde com saída sem sentido
  do modelo base — **não use nenhuma das duas**.
* **Sem instrução por requisição:** o reranker entende instruções (ganho típico de +1–5%), mas o
  llama.cpp aplica a instrução embutida no template de rerank do GGUF (o padrão oficial:
  *"Given a web search query, retrieve relevant passages that answer the query"*). A API não tem
  campo de instrução; personalizá-la significa reescrever os metadados do GGUF ou reconverter o
  modelo.
* **Fonte dos pesos:** a equipe Qwen não publica GGUFs oficiais para os rerankers, e **a maioria
  dos GGUFs da comunidade é quebrada para *reranking*** (convertidos como LMs causais simples,
  retornam scores sem sentido — veja
  [llama.cpp#16407](https://github.com/ggml-org/llama.cpp/issues/16407)). Este snap fixa as
  conversões [Voodisss `-llama_cpp`](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)
  (conversor oficial do llama.cpp, metadados completos de reranker), verificadas por sha256, com
  scores validados contra os valores de referência oficiais do model card. Não troque a fonte
  dos GGUFs sem rodar novamente `make rerank-test`.
* **Pin de runtime:** o llama.cpp está fixado em `b10344` — o pin da família (minicpm5-2b,
  qwen3-embedding, qwen3guard). O suporte ao Qwen3-Reranker exige builds mais novos que o
  [PR #15824](https://github.com/ggml-org/llama.cpp/pull/15824) (2025-09-25). Atualizações
  chegam como PRs revisados do Renovate (nunca automerge) e precisam passar na bateria E2E
  versionada (`tests/rerank-e2e.py`).
* **Quantização:** Q8_0 para 0.6B/4B (quase sem perdas; −0,2/−0,3% de NDCG segundo o benchmark
  do conversor) e Q4_K_M para 8B (a maior quantização que cabe no limite de 5 GiB por componente
  da loja; dentro da margem de ruído do F16). Evite qualquer coisa abaixo de Q4_K_M para
  rerankers (o Q2_K perde 4–29%).
* **Licença do modelo:** Apache-2.0 (equipe Qwen, Alibaba Group). A conversão GGUF (Voodisss) é
  Apache-2.0. O código de empacotamento é GPL-3.0.
* **Seleção de motor:** os motores `nvidia-gpu` e `amd-gpu` são escolhidos sempre que
  existe uma GPU do fornecedor correspondente no barramento PCI, mesmo que o driver não
  esteja utilizável. Nesse caso o backend falha ao inicializar e o llama.cpp cai para o
  backend de CPU; rode `sudo qwen3-reranker use-engine cpu` para voltar e liberar o
  componente do motor (~1,1 GB no NVIDIA, ~3,8 GB no AMD). O motor AMD só casa as
  microarquiteturas suportadas pelo build ROCm embarcado (gfx1010–gfx1201, incluindo a
  RX 9070 XT); GPUs AMD mais antigas selecionam o motor cpu. A seleção roda na instalação e
  precisa da interface `hardware-observe`; se a auto-conexão não tiver sido concedida,
  conecte e rode a seleção manualmente:
  ```shell
  sudo snap connect qwen3-reranker:hardware-observe
  sudo qwen3-reranker use-engine --auto
  ```
* **Provider:** o snap exporta um slot de conteúdo `inference-provider`, para que outro snap
  possa consumir este endpoint em vez de rodar o próprio reranker.
* **API local:** o endpoint não tem autenticação e permite requisições de outras origens. Ele
  escuta apenas em `127.0.0.1` — não rebinda `http.host` para uma interface pública.

## Compilar e instalar a partir do código

Clone o repositório:
```shell
git clone https://github.com/amenezes/qwen3reranker
cd qwen3reranker
```

Inicialize o ambiente de desenvolvimento:
```shell
make init        # apenas os pesos padrão 0.6B (~0,64 GB)
make init-all    # os três tamanhos (~9,6 GB) — necessário antes do `make build`
```

O `snapcraft` compila dentro de um contêiner por padrão, então um provider de build precisa
estar disponível antes. Com LXD, inicialize uma vez (`--storage-backend=dir` evita um pool ZFS
em loop):

```shell
lxd init --auto --storage-backend=dir
```

Sem provider, compile no host com `sudo snapcraft pack --destructive-mode --platform amd64`.

Compile e instale o snap:
```shell
make build
make install
```

Teste o snap instalado:
```shell
make smoke-test                                  # fundamentos do framework (motor, configuração, porta)
make rerank-test                                 # bateria E2E de 16 verificações no modelo padrão
make rerank-test MODEL=qwen3-reranker-8b         # após `use-model qwen3-reranker-8b`
```

Para usar uma cópia local dos pesos, coloque os arquivos em `components/` conforme documentado
no `README.md` de cada componente e rode `make verify-model` / `make verify-extra-models`.

## Publicação na Snap Store

Dois pedidos do lado da loja são pré-requisitos e ambos demoram, então abra-os cedo:

* **Acesso a upload de componentes** — os pesos e o runtime llama.cpp são distribuídos como
  [components](https://snapcraft.io/docs/explanation/how-snaps-work/snap-components/), e o
  upload de componentes é limitado a contas na allow-list da loja. Peça no
  [fórum do Snapcraft](https://forum.snapcraft.io/c/store-requests/).
* **Auto-conexão de `hardware-observe`** — a interface não é auto-conectada por padrão, então
  uma instalação pela loja não detecta o hardware no `use-engine --auto` até o pedido ser
  concedido (veja o
  [processo](https://snapcraft.io/docs/reference/administration/process-for-aliases-auto-connections-and-tracks/)).

O primeiro upload também exige um nome registrado e um repositório que exista de fato: a página
da loja liga `website`, `source-code`, `issues` e `contact` do `snap/snapcraft.yaml`.

```shell
snapcraft login
snapcraft register qwen3-reranker
make build
./dev/upload.sh latest/edge   # snapcraft upload <snap> --component ... --release=latest/edge
```

`.github/workflows/cicd.yaml` publica em `latest/edge` a cada push na `main` assim que o secret
`SNAPCRAFT_STORE_CREDENTIALS` (saída de `snapcraft export-login`) estiver configurado.
