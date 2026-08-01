"""
Full-scale (H100) configuration from the original notebook.

This is the production target the learning notebook is heading toward.
Use it as a reference when you are ready to move off laptop hardware.
"""

FULL_SCALE_CONFIG = {
    # Infrastructure
    "hardware": "NVIDIA H100 80GB",
    "ram_gb": 180,
    "disk": "750GB NVMe",
    "generator_endpoint": "http://localhost:8000/v1",  # warm vLLM server

    # Run profile
    "SMOKE_TEST": False,
    "SLICE_SIZE": 50_000,           # target corpus passages
    "N_EVAL_ANSWERABLE": 300,
    "N_EVAL_UNANSWERABLE": 300,
    "RUN_SCALE_LAB": True,
    "CONTEXTUALIZE_LIMIT": 0,       # 0 means no limit

    # Models (all local)
    "GEN_MODEL": "Qwen/Qwen3-32B",
    "EMBED_MODEL_OFFLINE": "Qwen/Qwen3-Embedding-8B",
    "EMBED_MODEL_ONLINE": "Qwen/Qwen3-Embedding-4B",
    "RERANK_MODEL": "Qwen/Qwen3-Reranker-4B",
    "VERIFIER_MODEL": "bespokelabs/Bespoke-MiniCheck-7B",
    "NLI_MODEL": "cross-encoder/nli-deberta-v3-base",
    "USE_LLM_JUDGE": True,

    # Retrieval / agent knobs
    "CHUNK_TOKENS": 256,
    "CHUNK_OVERLAP": 32,
    "RETRIEVE_K": 150,              # fused candidates before rerank
    "RERANK_TOP_N": 20,             # passages handed to the generator
    "RRF_K": 60,
    "MAX_HOPS": 3,                  # agent corrective-loop cap
    "CRAG_OK": 0.70,                # >= -> evidence sufficient
    "CRAG_BAD": 0.40,               # <  -> abstain without generating
    "TAU_CLAIM": 0.30,              # per-claim support threshold
    "TAU_ABSTAIN": 0.30,            # final abstention threshold
    "THINKING_MODE": False,

    # Scale-lab sizes (literal 10M-vector index benchmark)
    "SCALE_DIM": 1024,
    "SCALE_SIZES": [100_000, 1_000_000, 10_000_000],
    "SCALE_TARGET": 100_000_000,    # projection point

    # Paths
    "DATA_DIR": "/mnt/data",
    "SCRATCH_DIR": "/mnt/scratch",
    "SEED": 42,
}
