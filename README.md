# Building a RAG Pipeline for 10M+ Documents With Near-Zero Hallucination

A hands-on learning repo that rebuilds an agentic RAG pipeline from scratch, step by step, using small models that run on a laptop.

This is a lightweight, self-paced version of the full H100 project in `../rag-zero-hallucinations`. The full project uses `Qwen3-32B`, `Qwen3-Embedding-8B`, and a single NVIDIA H100 80GB GPU. This learning edition uses `Qwen2.5-1.5B-Instruct`, `all-MiniLM-L6-v2`, and your CPU (or a small GPU) so you can learn the architecture without cloud costs.

## What you'll build

A complete RAG agent that:

1. **Cleans and chunks** a document corpus.
2. **Indexes** chunks with dense vectors (LanceDB) + sparse BM25.
3. **Retrieves** with hybrid search + reciprocal rank fusion + reranking.
4. **Routes and decomposes** questions.
5. **Generates** answers strictly from retrieved context with inline citations.
6. **Verifies** every atomic claim with an NLI model.
7. **Abstains** when evidence is insufficient.
8. **Evaluates** itself with a 2×2 confusion matrix and a risk-coverage curve.

## Quick start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-learning.txt

# 3. Launch Jupyter
jupyter notebook notebooks/rag_zero_hallucination_learning.ipynb
```

## How to learn this

Don't run the whole notebook at once. Use `LEARNING_PATH.md` as your guide. It breaks the notebook into 12 focused stages, each with:

- The relevant notebook section.
- Key concepts.
- A check-your-understanding question.
- Suggested timing.

## Repository layout

```
.
├── notebooks/
│   └── rag_zero_hallucination_learning.ipynb   # the hands-on notebook
├── scale/                                      # production-scale reference code
│   ├── config_full_scale.py                    # H100 target configuration
│   ├── scale_lab.py                            # 100K -> 1M -> 10M vector benchmark
│   ├── contextual_retrieval.py                 # LLM-powered chunk contextualization
│   ├── latency_dashboard.py                    # per-stage latency & cost aggregation
│   └── README.md                               # scale artifacts guide
├── LEARNING_PATH.md                            # step-by-step study guide
├── requirements-learning.txt                   # CPU-friendly dependencies
└── README.md                                   # this file
```

## Hardware expectations

- **Minimum:** 8 GB RAM, CPU only.
- **Recommended:** 16 GB RAM, or any GPU with 8 GB+ VRAM.
- **First run:** downloads ~2–4 GB of models from Hugging Face.
- **Runtime:** 10–30 minutes for the default tiny config on CPU.

## Scaling up

Once you understand the learning version, the jump to the full pipeline is mostly model and corpus swaps:

| Component | Learning | Full scale |
|-----------|----------|------------|
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen3-32B` via vLLM |
| Embeddings | `all-MiniLM-L6-v2` | `Qwen/Qwen3-Embedding-8B` |
| Reranker | `ms-marco-MiniLM-L-6-v2` | `Qwen/Qwen3-Reranker-4B` |
| Verifier | `nli-deberta-v3-base` | `Bespoke-MiniCheck-7B` + LLM judge |
| Corpus | 500 passages | 20K–50K+ passages |
| Scale lab | skipped | real 100K → 1M → 10M vector index |
| Hardware | laptop CPU | NVIDIA H100 80GB |

## Scale reference code

The `scale/` directory contains the production-scale pieces extracted from the full H100 notebook. Use these when you are ready to move beyond laptop experiments:

- **`scale/config_full_scale.py`** — full H100 config: `Qwen3-32B`, `Qwen3-Embedding-8B`, `Bespoke-MiniCheck-7B`, corpus sizes, thresholds, and hardware assumptions.
- **`scale/scale_lab.py`** — literally builds LanceDB indexes at 100K / 1M / 10M vectors, measures build time, on-disk size, p50/p95 query latency, and recall@10, then projects to 100M.
- **`scale/contextual_retrieval.py`** — Anthropic-style contextual retrieval that prepends an LLM-written situating sentence to every chunk before indexing.
- **`scale/latency_dashboard.py`** — aggregates per-stage latencies and builds a headline metrics card for cost analysis.

These are reference implementations. You will need to adapt them to your own `Chunk`, corpus, and LLM client abstractions.

## Key papers & ideas behind this pipeline

- **RAG:** Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020).
- **BM25:** Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- **Reciprocal Rank Fusion:** Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods* (2009).
- **Contextual Retrieval:** Anthropic (2024) — prepending situating context to chunks.
- **Self-RAG / CRAG:** Asai et al., *Self-RAG* (2023) and Corrective RAG patterns.
- **Chain-of-Verification:** Dhuliawala et al., *Chain-of-Verification Reduces Hallucination in Large Language Models* (2023).
- **HaluBench:** Wang et al., *HaluEval: A Large-Scale Hallucination Evaluation Benchmark* (2023).

## License

Same as the source project. See `LICENSE`.
