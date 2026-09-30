#!/usr/bin/env python3
"""E2E test battery for the qwen3-reranker snap (Jina/Cohere-style /v1/rerank API).

Runs against any llama-server in rerank mode serving a Qwen3-Reranker GGUF
converted with the official llama.cpp converter -- either the installed snap
(default: port 8366) or a bare llama-server started from the pinned runtime
for pre-build validation.

Usage:
    python3 tests/rerank-e2e.py [BASE_URL] [MODEL_NAME] [MAX_PROMPT_TOKENS]

    BASE_URL           default http://127.0.0.1:8366
    MODEL_NAME         default qwen3-reranker-0-6b (the OpenAI model id / alias)
    MAX_PROMPT_TOKENS  largest prompt (query+template+doc) the server accepts;
                       in rerank mode llama-server clamps n_batch to n_ubatch,
                       so this is the engine's ubatch-size (default: derive
                       from /props n_ctx, correct when ubatch-size >= ctx).

Only the Python standard library is used, so it also runs inside minimal
environments (CI runners, containers) without setup.

Reference scores: the official model cards publish raw logit differences for
the (Beijing, gravity) example -- 0.6B: [7.625, -11.375], 4B: [6.4375,
-14.375], 8B: [5.0625, -14.25]. llama.cpp reports the 2-way softmax over the
yes/no logits, i.e. sigmoid(logit difference): >= 0.9937 for the smallest of
the three on the relevant doc. The battery asserts > 0.98 / < 0.01, which all
three sizes clear with margin, and prints the exact expected sigmoid for
comparison.
"""
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8366"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3-reranker-0-6b"
MAX_PROMPT = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = derive from n_ctx
results = []

QUERY = "What is the capital of China?"
DOC_RELEVANT = "The capital of China is Beijing."
DOC_IRRELEVANT = (
    "Gravity is a force that attracts two bodies towards each other. It gives "
    "weight to physical objects and is responsible for the movement of planets "
    "around the sun."
)
# Official model-card raw logit differences -> expected sigmoid (P("yes")).
REFERENCE_SIGMOID = {"0-6b": 0.99951, "4b": 0.99840, "8b": 0.99368}


def request(path, payload=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body.decode()[:400]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def rerank(query, documents, **extra):
    status, body = request("/v1/rerank",
                           {"model": MODEL, "query": query,
                            "documents": documents, **extra})
    if status != 200:
        raise AssertionError(f"HTTP {status}: {body}")
    return body


def scores_by_index(body):
    return {r["index"]: r["relevance_score"] for r in body["results"]}


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------- T1 health
def t1_health():
    status, _ = request("/health")
    check("T1 /health is 200", status == 200, f"status={status}")


# ----------------------------------------------------------------- T2 models
def t2_models():
    status, body = request("/v1/models")
    ids = [m["id"] for m in body.get("data", [])] if status == 200 else []
    check("T2 /v1/models serves the alias", status == 200 and MODEL in ids,
          f"ids={ids}")


# --------------------------------------------- T3 official reference scores
def t3_reference_scores():
    body = rerank(QUERY, [DOC_RELEVANT, DOC_IRRELEVANT])
    s = scores_by_index(body)
    key = next((v for k, v in REFERENCE_SIGMOID.items() if k in MODEL), 0.99951)
    ok = (s.get(0, -1) > 0.98 and s.get(1, 1) < 0.01 and s[0] > s[1])
    check("T3 model-card reference scores", ok,
          f"relevant={s.get(0):.6g} (expected ~{key:.5f}) "
          f"irrelevant={s.get(1):.6g} (expected ~{sigmoid(-11.375):.3g})")


# ------------------------------------------------------- T4 sorted + top_n
def t4_sorted_top_n():
    docs = [DOC_IRRELEVANT, DOC_RELEVANT, "Paris is the capital of France.",
            "The Great Wall is one of the most famous landmarks in China."]
    body = rerank(QUERY, docs)
    rs = [r["relevance_score"] for r in body["results"]]
    sorted_desc = all(a >= b for a, b in zip(rs, rs[1:]))
    top_is_relevant = body["results"][0]["index"] == 1
    body2 = rerank(QUERY, docs, top_n=2)
    top2 = len(body2["results"]) == 2 and body2["results"][0]["index"] == 1
    check("T4 results sorted desc, top hit relevant, top_n truncates",
          sorted_desc and top_is_relevant and top2,
          f"order={[r['index'] for r in body['results']]} "
          f"top_n2={[r['index'] for r in body2['results']]}")


# ------------------------------------------------- T5 semantic discrimination
def t5_discrimination():
    query = "Qual é a capital do Brasil?"
    docs = [
        "The weather in Paris is rainy during the spring.",
        "Brasília is the federal capital of Brazil, located in the Central-West region.",
        "O futebol é o esporte mais popular do Brasil.",
    ]
    body = rerank(query, docs)
    top = body["results"][0]["index"]
    # Cross-lingual (pt query, en doc) must win over an in-language distractor.
    check("T5 pt->en cross-lingual: relevant doc ranked 1st", top == 1,
          f"top_index={top} scores={scores_by_index(body)}")


# ------------------------------------------------------------ T6 score range
def t6_score_range():
    docs = [DOC_RELEVANT, DOC_IRRELEVANT, "", "x", QUERY]
    body = rerank(QUERY, docs)
    s = [r["relevance_score"] for r in body["results"]]
    check("T6 all scores in [0, 1]", all(0.0 <= x <= 1.0 for x in s),
          f"scores={['%.4g' % x for x in s]}")


# ------------------------------------------------------------ T7 bad input
def t7_bad_input():
    st1, b1 = request("/v1/rerank", {"model": MODEL, "query": QUERY,
                                     "documents": []})
    st2, b2 = request("/v1/rerank", {"model": MODEL,
                                     "documents": [DOC_RELEVANT]})
    st3, b3 = request("/v1/rerank", {"model": MODEL, "query": 42,
                                     "documents": [DOC_RELEVANT]})
    check("T7 invalid requests rejected with 4xx",
          st1 == 400 and st2 == 400 and st3 == 400,
          f"empty_docs={st1} missing_query={st2} non_string_query={st3}")


# ------------------------------------------------------------- T8 TEI format
def t8_tei_format():
    status, body = request("/v1/rerank",
                           {"model": MODEL, "query": QUERY,
                            "texts": [DOC_IRRELEVANT, DOC_RELEVANT],
                            "return_text": True})
    ok = (status == 200 and isinstance(body, list) and len(body) == 2
          and "score" in body[0] and body[0]["index"] == 1
          and body[0].get("text") == DOC_RELEVANT)
    check("T8 TEI format ('texts' + return_text)", ok,
          f"status={status} first={str(body[0] if isinstance(body, list) and body else body)[:120]}")


# ----------------------------------------------------------- T9 long input
def t9_long_input():
    # Context-aware: the bare pre-build server runs with -c 8192, the snap fits
    # the context to available memory (4096 floor, up to 40960). Ask /props.
    status, props = request("/props")
    n_ctx = 8192
    if status == 200 and isinstance(props, dict):
        n_ctx = int(props.get("default_generation_settings", {}).get("n_ctx", n_ctx))
    limit = MAX_PROMPT if MAX_PROMPT else n_ctx  # processable prompt tokens
    long_doc = "The history of China spans thousands of years. " * max(
        10, int(min(n_ctx, limit) * 0.75) // 9)  # ~9 tokens per repetition
    status, body = request("/v1/rerank", {"model": MODEL, "query": QUERY,
                                          "documents": [long_doc]})
    ok_long = status == 200 and isinstance(body, dict) and len(body["results"]) == 1
    huge_doc = "word " * int(n_ctx * 1.5)  # > served context
    st2, b2 = request("/v1/rerank", {"model": MODEL, "query": QUERY,
                                     "documents": [huge_doc]})
    ok_huge = st2 >= 400  # must be rejected clearly, never silently truncated
    check("T9 long doc scored; oversized doc rejected (no silent truncation)",
          ok_long and ok_huge,
          f"n_ctx={n_ctx} limit={limit} long={status} oversized={st2}")


# --------------------------------------- T10 non-rerank surfaces are dead
def t10_dead_surfaces():
    # /v1/embeddings answers but is degenerate on a reranker: b10344 returns a
    # hidden-dim vector that is all zeros (0.6B) or near-all zeros with <=2
    # stray values (4B/8B), varying with server state. Pin "degenerate" (<=1%
    # nonzero) rather than exact zeros, so a runtime bump that turns the
    # surface into a real embedding API (or refuses it) is caught by the
    # battery. /v1/chat/completions is NOT refused by b10344 in rerank mode;
    # it replies with base-model nonsense -- a footgun documented in the
    # README instead of asserted here.
    status, body = request("/v1/embeddings",
                           {"model": MODEL, "input": [DOC_RELEVANT]})
    degenerate = False
    nz = dim = -1
    if status == 200 and isinstance(body, dict):
        vec = body["data"][0]["embedding"]
        dim = len(vec)
        nz = sum(1 for x in vec if x != 0.0)
        degenerate = dim > 0 and nz <= max(2, dim // 100)
    ok_rerank = len(rerank(QUERY, [DOC_RELEVANT])["results"]) == 1
    check("T10 embeddings surface degenerate; rerank unaffected",
          degenerate and ok_rerank,
          f"embeddings_status={status} dim={dim} nonzero={nz}")


# ------------------------------------------------------ T11 batch of ten
def t11_batch():
    docs = [f"Document number {i} about topic {i % 3}." for i in range(10)]
    docs[7] = DOC_RELEVANT
    body = rerank("What is the capital of China?", docs)
    s = scores_by_index(body)
    ok = (len(s) == 10 and set(s) == set(range(10))
          and body["results"][0]["index"] == 7)
    check("T11 batch of 10: all scored, relevant wins", ok,
          f"n={len(s)} top={body['results'][0]['index']}")


# ------------------------------------------------------- T12 concurrency
def t12_concurrency():
    out = {}

    def call(i):
        try:
            out[i] = rerank(QUERY, [DOC_RELEVANT, DOC_IRRELEVANT])
        except Exception as e:  # noqa: BLE001
            out[i] = e

    ts = [threading.Thread(target=call, args=(i,)) for i in range(2)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=600)
    ok = all(isinstance(v, dict) for v in out.values()) and len(out) == 2
    check("T12 two concurrent requests both succeed", ok,
          f"elapsed={time.time() - t0:.1f}s")


# --------------------------------------------------------- T13 usage tokens
def t13_usage():
    body = rerank(QUERY, [DOC_RELEVANT])
    u = body.get("usage", {})
    check("T13 usage.prompt_tokens reported", u.get("prompt_tokens", 0) > 0,
          f"usage={u}")


# ------------------------------------------------------ T14 determinism
def t14_determinism():
    a = rerank(QUERY, [DOC_RELEVANT, DOC_IRRELEVANT])
    b = rerank(QUERY, [DOC_RELEVANT, DOC_IRRELEVANT])
    check("T14 deterministic scores", a["results"] == b["results"],
          f"run1={scores_by_index(a)} run2={scores_by_index(b)}")


# ----------------------------------- T15 unknown fields ignored (no instruct)
def t15_no_instruct_field():
    # llama.cpp's /v1/rerank has no instruct/prompt field: the instruction is
    # baked into the GGUF's rerank template. Unknown fields must be ignored,
    # i.e. scores identical with and without a client-sent "instruct".
    a = scores_by_index(rerank(QUERY, [DOC_RELEVANT, DOC_IRRELEVANT]))
    b = scores_by_index(rerank(QUERY, [DOC_RELEVANT, DOC_IRRELEVANT],
                               instruct="Classify if the document answers the query"))
    check("T15 client 'instruct' field ignored (template-baked instruction)",
          a == b, f"without={a} with={b}")


# ------------------------------------------------- T16 throughput (info only)
def t16_throughput():
    docs = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 8] * 8
    t0 = time.time()
    body = rerank("summarize this text", docs)
    dt = time.time() - t0
    n_tok = body.get("usage", {}).get("total_tokens", 0)
    results.append(("T16 throughput", True,
                    f"{len(docs)} docs in {dt:.2f}s ({len(docs) / dt:.1f} docs/s, "
                    f"{n_tok / dt:.0f} tok/s prefill)"))
    print(f"INFO  T16 throughput  {len(docs)} docs in {dt:.2f}s "
          f"({len(docs) / dt:.1f} docs/s, {n_tok / dt:.0f} tok/s prefill)")


def main():
    print(f"# rerank E2E battery: {BASE} model={MODEL}")
    for t in (t1_health, t2_models, t3_reference_scores, t4_sorted_top_n,
              t5_discrimination, t6_score_range, t7_bad_input, t8_tei_format,
              t9_long_input, t10_dead_surfaces, t11_batch, t12_concurrency,
              t13_usage, t14_determinism, t15_no_instruct_field,
              t16_throughput):
        try:
            t()
        except Exception as e:  # noqa: BLE001
            check(t.__name__, False, f"EXCEPTION {e!r}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        sys.exit(1)
    print("E2E OK")


if __name__ == "__main__":
    main()
