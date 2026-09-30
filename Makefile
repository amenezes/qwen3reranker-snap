SHELL := /bin/bash

# Always run `hf` via pipx to avoid relying on local `hf` installations.
hf := pipx run --spec "huggingface_hub[cli]" hf

SNAP_NAME ?= qwen3-reranker
ENGINE ?= cpu

# Model weights: GGUF conversions published by Voodisss with the official
# llama.cpp converter (convert_hf_to_gguf.py). The Qwen team publishes no
# official GGUF for the rerankers (unlike Qwen3-Embedding), and most community
# GGUFs (mradermacher, QuantFactory, Mungert, DevQuasar...) were converted as
# plain causal LMs: they lack the cls.output yes/no head, the RANK pooling
# metadata and the "rerank" chat template, so llama-server --rerank returns
# garbage scores with them (llama.cpp#16407). The Voodisss "-llama_cpp" repos
# are the only conversions carrying the full reranker metadata; their scores
# were verified against the official model-card reference (ai/03). 0.6B/4B in
# Q8_0 (near-lossless, fits the 5 GiB component limit); 8B in Q4_K_M because
# its Q8_0 (8.0 GB) and Q5_K_M (5.4 GB) exceed the limit. Checksums pinned
# from the HF LFS metadata of 2026-09-30 (see components/*/README.md and
# ai/07-decisoes.md).
GGUF_06B_REPO ?= Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp
GGUF_06B_FILE ?= Qwen3-Reranker-0.6B.Q8_0.gguf
GGUF_06B_SHA256 ?= 6ddab39a36c6c87fdb76f0e5f05657012d5dbc97034c0983c157f17ef9f34d55
MODEL_06B_DIR := components/model-0-6b-q8-0-gguf

GGUF_4B_REPO ?= Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp
GGUF_4B_FILE ?= Qwen3-Reranker-4B.Q8_0.gguf
GGUF_4B_SHA256 ?= 102ac400d68f02877c2fdfecf0732872652298448352b6faf28ec7f8dca8c913
MODEL_4B_DIR := components/model-4b-q8-0-gguf

GGUF_8B_REPO ?= Voodisss/Qwen3-Reranker-8B-GGUF-llama_cpp
GGUF_8B_FILE ?= Qwen3-Reranker-8B-Q4_K_M.gguf
GGUF_8B_SHA256 ?= b0de7188a5801565d87c895d4ee8b6a1da3f46de3f07f378b831097894f210d2
MODEL_8B_DIR := components/model-8b-q4-k-m-gguf

.PHONY: all help init init-all build install upload smoke-test rerank-test install-deps init-submodules download-models download-extra-models verify-model verify-extra-models

all: help

#
# Main targets
#

help: ## Show this help message
	@echo "Usage: make <target>"
	@echo
	@echo "Targets:"
	@# List all targets with descriptions (lines starting with '##'):
	@grep -E '^[a-zA-Z0-9_-]+:.*## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

init: init-submodules install-deps download-models verify-model ## Initialize the build environment (default 0.6B model only)

init-all: init-submodules install-deps download-models download-extra-models verify-model verify-extra-models ## Initialize with all model sizes (0.6B, 4B, 8B)

build: ## Build the snap (needs all model files; use init-all first)
	./dev/build.sh

install: ## Install the snap
	./dev/install.sh

upload: ## Upload the snap
	./dev/upload.sh

smoke-test: ## Run smoke tests (override with SNAP_NAME=... ENGINE=...)
	sudo ./dev/smoke-test.sh $(SNAP_NAME) $(ENGINE)

rerank-test: ## Run the rerank E2E battery against a running server (PORT=..., MODEL=..., UBATCH=...)
	python3 tests/rerank-e2e.py http://127.0.0.1:$(or $(PORT),8366) $(or $(MODEL),qwen3-reranker-0-6b) $(or $(UBATCH),2048)

#
# Supporting targets
#

install-deps:
	@echo "Installing dependencies..."
	@# Ensure pipx is available for running the hf CLI.
	@command -v pipx >/dev/null 2>&1 || { \
		sudo apt-get update; \
		sudo apt-get install -y pipx; \
	}

init-submodules:
	@echo "Initializing submodules..."
	@if git submodule status | grep -q '^-'; then \
		git submodule update --init; \
	fi

download-models:
	@echo "Downloading Qwen3-Reranker-0.6B-Q8_0 model weights (~0.6 GB)..."
	$(hf) download $(GGUF_06B_REPO) $(GGUF_06B_FILE) \
		--local-dir $(MODEL_06B_DIR)

download-extra-models: ## Download the 4B and 8B model weights (~9.0 GB total)
	@echo "Downloading Qwen3-Reranker-4B-Q8_0 model weights (~4.3 GB)..."
	$(hf) download $(GGUF_4B_REPO) $(GGUF_4B_FILE) \
		--local-dir $(MODEL_4B_DIR)
	@echo "Downloading Qwen3-Reranker-8B-Q4_K_M model weights (~5.0 GB)..."
	$(hf) download $(GGUF_8B_REPO) $(GGUF_8B_FILE) \
		--local-dir $(MODEL_8B_DIR)

verify-model: ## Verify the 0.6B GGUF checksum against the pinned sha256
	@echo "Verifying $(MODEL_06B_DIR)/$(GGUF_06B_FILE)..."
	@if [ ! -f "$(MODEL_06B_DIR)/$(GGUF_06B_FILE)" ]; then \
		echo "ERROR: $(MODEL_06B_DIR)/$(GGUF_06B_FILE) not found. Run 'make download-models' first." >&2; \
		exit 1; \
	fi
	@echo "$(GGUF_06B_SHA256)  $(MODEL_06B_DIR)/$(GGUF_06B_FILE)" | sha256sum --check --status \
		&& echo "OK: 0.6B checksum matches" \
		|| { echo "ERROR: 0.6B checksum mismatch (expected $(GGUF_06B_SHA256))" >&2; exit 1; }

verify-extra-models: ## Verify the 4B and 8B GGUF checksums
	@echo "Verifying $(MODEL_4B_DIR)/$(GGUF_4B_FILE)..."
	@if [ ! -f "$(MODEL_4B_DIR)/$(GGUF_4B_FILE)" ]; then \
		echo "ERROR: $(MODEL_4B_DIR)/$(GGUF_4B_FILE) not found. Run 'make download-extra-models' first." >&2; \
		exit 1; \
	fi
	@echo "$(GGUF_4B_SHA256)  $(MODEL_4B_DIR)/$(GGUF_4B_FILE)" | sha256sum --check --status \
		&& echo "OK: 4B checksum matches" \
		|| { echo "ERROR: 4B checksum mismatch (expected $(GGUF_4B_SHA256))" >&2; exit 1; }
	@echo "Verifying $(MODEL_8B_DIR)/$(GGUF_8B_FILE)..."
	@if [ ! -f "$(MODEL_8B_DIR)/$(GGUF_8B_FILE)" ]; then \
		echo "ERROR: $(MODEL_8B_DIR)/$(GGUF_8B_FILE) not found. Run 'make download-extra-models' first." >&2; \
		exit 1; \
	fi
	@echo "$(GGUF_8B_SHA256)  $(MODEL_8B_DIR)/$(GGUF_8B_FILE)" | sha256sum --check --status \
		&& echo "OK: 8B checksum matches" \
		|| { echo "ERROR: 8B checksum mismatch (expected $(GGUF_8B_SHA256))" >&2; exit 1; }
