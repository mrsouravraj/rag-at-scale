# Scale artifacts

This directory holds the **production-scale** reference code and configuration
extracted from the full H100 notebook in `../rag-zero-hallucinations`.

The learning notebook (`notebooks/rag_zero_hallucination_learning.ipynb`) runs the
same architecture on small models and a tiny corpus. The files here show what
changes when you move to real scale.

## Files

| File | What it is |
|------|------------|
| `config_full_scale.py` | H100 target configuration: model names, corpus sizes, thresholds, hardware assumptions. |
| `scale_lab.py` | Builds LanceDB indexes from **real corpus embeddings** at 100k / 1M / 10M vectors and measures build time, disk, p50/p95 latency, and recall@10. No synthetic vectors. |
| `contextual_retrieval.py` | Anthropic-style contextual retrieval: prepend an LLM-written one-line context to every chunk before indexing. |
| `latency_dashboard.py` | Aggregate per-stage latencies and build a headline metrics card. |

## When to use these

- **Local learning:** start with the notebook and `requirements-learning.txt`.
- **First scale-up:** compare your working config against `config_full_scale.py`.
- **Indexing at scale:** adapt `scale_lab.py` to your vector dimension and ANN parameters.
- **Cost/latency analysis:** wire `latency_dashboard.py` into your agent loop.

## Note

These modules are extracted reference implementations. They assume you have the
surrounding `Chunk`, corpus, and LLM client objects available. Treat them as a
starting point, not a drop-in executable.
