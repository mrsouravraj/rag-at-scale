# Learning Path: Build a Near-Zero-Hallucination RAG Agent

This repository contains a single hands-on notebook that rebuilds an agentic RAG pipeline from scratch, using small models so it can run on a laptop.

**Notebook:** `notebooks/rag_zero_hallucination_learning.ipynb`

---

## How to use this guide

Don't run the whole notebook at once. Work through the numbered stages below. For each stage:

1. Read the markdown cell(s) first — they explain the concept.
2. Run the code cells in that section.
3. Answer the "Check your understanding" question in your head or in a notes file.
4. Only move on once you can explain what that stage does and why it matters.

Expected total time: 2–4 hours spread over several sessions.

---

## Stage 0 — Setup & first run (10 min)

**Cells:** top of notebook through the config / GPU check.

**What you'll do:**
- Install dependencies (`pip install -r requirements-learning.txt`).
- Open the notebook and run the first few cells.
- Confirm the small-model config is active (`Qwen2.5-1.5B-Instruct`, `all-MiniLM-L6-v2`, `SLICE_SIZE=500`).

**Key ideas:**
- Why we shrink the corpus and models for learning.
- How `DATA_DIR` and `SCRATCH_DIR` are local project folders, not `/mnt/data`.
- The difference between the original H100 setup and this learning setup.

**Check:** What does `LEARNING_MODE = True` change in the notebook?

---

## Stage 1 — Understand the data (15 min)

**Cells:** Section 1 (Download datasets & build corpus).

**What you'll do:**
- Load HotpotQA.
- Build a small passage corpus.
- Inspect one question and its gold evidence.

**Key ideas:**
- What is a "passage" vs a "chunk"?
- What are "gold titles" and why do we need them?
- What makes a question "multi-hop"?

**Check:** Why is it important to know which passages *should* have been retrieved?

---

## Stage 2 — Text cleaning & near-duplicate detection (15 min)

**Cells:** Section 2 (Parse / normalize / dedup).

**What you'll do:**
- Normalize Unicode and whitespace.
- Run MinHash LSH to drop near-duplicate passages.
- Compare counts before and after.

**Key ideas:**
- Why normalization matters for BM25 tokenization.
- How near-duplicates pollute retrieval metrics.
- What MinHash LSH approximates and why it's fast.

**Check:** If two passages are 95% identical, why is keeping both harmful?

---

## Stage 3 — Structure-aware chunking (15 min)

**Cells:** Section 3 (Structure-aware chunking).

**What you'll do:**
- Split passages into sentence-preserving chunks.
- See chunk token statistics.

**Key ideas:**
- Why fixed-size chunking can break entity context.
- Why we use the generator's tokenizer to count tokens.
- The role of chunk overlap.

**Check:** What bad thing happens if a person's name is in one chunk and the fact about them is in another?

---

## Stage 4 — Hybrid retrieval: dense + sparse (30 min)

**Cells:** Sections 5–7 (embedder, LanceDB, BM25, RRF, reranking).

**What you'll do:**
- Load the sentence transformer embedder.
- Build a dense vector index in LanceDB.
- Build a sparse BM25 index.
- Fuse both rankings with Reciprocal Rank Fusion.
- Rerank top candidates with a small cross-encoder.
- Measure passage recall vs gold titles.

**Key ideas:**
- What dense embeddings are good at (paraphrase, semantics).
- What BM25 is good at (exact tokens, names, IDs).
- Why fusion beats either method alone.
- Why reranking is a second, more expensive precision stage.

**Check:** For the query "Were Scott Derrickson and Ed Wood of the same nationality?", which index likely finds "Ed Wood" first — dense or BM25?

---

## Stage 5 — Query routing & decomposition (15 min)

**Cells:** Section 8 (Query routing & decomposition).

**What you'll do:**
- Classify questions as `no_retrieval`, `single_hop`, or `multi_hop`.
- Decompose a multi-hop question into sub-questions.
- Detect false-premise questions.

**Key ideas:**
- Why not every question needs retrieval.
- How decomposition enables multi-hop reasoning.
- What a false-premise question looks like.

**Check:** If a question contains a false premise, is the correct behavior to answer it or abstain?

---

## Stage 6 — Firewall 1: constrained, cited generation (20 min)

**Cells:** Section 9 (Constrained, cited generation).

**What you'll do:**
- Format retrieved chunks as context.
- Prompt the generator to answer only from context and cite chunk IDs.
- Parse and validate citations, stripping hallucinated chunk IDs.

**Key ideas:**
- Why "answer only from context" reduces hallucination.
- What an abstain token is and when to emit it.
- Why we validate that cited IDs actually exist in the retrieved set.

**Check:** The model outputs `[abc123]`. What should happen if `abc123` was not in the retrieved chunks?

---

## Stage 7 — Firewall 2: atomic-claim verification (30 min)

**Cells:** Section 10 (Atomic-claim verification gate).

**What you'll do:**
- Split an answer into atomic claims.
- Use an NLI model to score `context → claim` entailment.
- Decide if the answer passes the gate.

**Key ideas:**
- What an "atomic claim" is.
- What NLI (natural language inference) means in this context.
- Why the lowest claim score is the bottleneck.
- The relationship between `TAU_CLAIM` and abstention.

**Check:** If one claim scores 0.9 and another scores 0.2, does the answer pass? What does that tell you?

---

## Stage 8 — Abstention policy (15 min)

**Cells:** Section 11 (Abstention policy & structured output).

**What you'll do:**
- Combine signals: router label, model abstain, verification gate.
- Produce a structured final answer or abstention.

**Key ideas:**
- Why abstention is a safe failure mode.
- How multiple signals are folded into one decision.
- Why structured output helps evaluation.

**Check:** List three different reasons the system might abstain.

---

## Stage 9 — The agent loop (20 min)

**Cells:** Section 12 (LangGraph CRAG / Self-RAG loop).

**What you'll do:**
- Wire the components into a LangGraph state machine.
- See route → retrieve → grade → refine → generate → verify → finalize.
- Run one question end-to-end.

**Key ideas:**
- What CRAG (Corrective RAG) does when evidence is weak.
- Why the loop has a `MAX_HOPS` cap.
- How generation is skipped entirely when evidence is too weak.

**Check:** Why is skipping generation on bad evidence better than generating anyway and then verifying?

---

## Stage 10 — Evaluation (30 min)

**Cells:** Sections 13–16 (Golden set, confusion matrix, risk-coverage, metrics, manifest).

**What you'll do:**
- Build a golden set of answerable and unanswerable questions.
- Run the agent over the set.
- Plot the 2×2 confusion matrix and risk-coverage curve.
- Read the headline metrics.

**Key ideas:**
- Why the dangerous cell is (unanswerable, answered).
- What the risk-coverage curve trades off.
- Why near-zero hallucination usually costs some coverage.

**Check:** If you increase `TAU_ABSTAIN`, what happens to coverage? What happens to hallucination rate?

---

## Stage 11 — Scale concepts (read-only, 10 min)

**Cells:** Section 15 (Scale lab) — skipped in learning mode.

**What you'll do:**
- Read the markdown.
- Look at the real measurements in the source repo (`results/scale_lab.json`).

**Key ideas:**
- What changes at 10M vectors vs 10K vectors (disk, ANN latency).
- What does NOT change (the correctness logic).
- Why scale is a separate concern from correctness.

**Check:** Why does the notebook not build a 10M index in learning mode?

---

## Suggested schedule

| Session | Stages | Time |
|---------|--------|------|
| 1 | 0–2 | 40 min |
| 2 | 3–4 | 45 min |
| 3 | 5–7 | 65 min |
| 4 | 8–10 | 80 min |
| 5 | 11 + free exploration | 30 min |

---

## After you finish

To scale up to the full project:

1. Switch models: `Qwen/Qwen3-32B` via vLLM, `Qwen3-Embedding-8B`, `Qwen3-Reranker-4B`, `Bespoke-MiniCheck-7B`.
2. Increase `SLICE_SIZE` to 20K–50K+.
3. Enable contextual retrieval (`RUN_CONTEXTUALIZE = True`).
4. Enable the scale lab (`RUN_SCALE_LAB = True`) on a machine with an NVIDIA GPU and ~80 GB VRAM.
5. See the original repo: `../rag-zero-hallucinations`.

---

## Common issues

- **First run downloads models:** Hugging Face will cache `~/.cache/huggingface` or `./learning_data/hf`. This is normal.
- **Out of memory:** Reduce `SLICE_SIZE`, `RETRIEVE_K`, `RERANK_TOP_N`, or use a smaller generator like `Phi-3-mini-4k-instruct`.
- **Slow on CPU:** A single question may take 1–3 minutes. For faster iteration, rent a GPU cloud instance.
