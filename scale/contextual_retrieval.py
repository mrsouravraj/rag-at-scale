"""
Contextual Retrieval: prepend a one-line, LLM-written situating context to each chunk.

Extracted from the original H100 notebook (Sec 4). This is the most expensive offline
step in the full pipeline (one short generation per chunk), but it can sharply improve
recall on isolated sentences like "revenue grew 3% that quarter".

Reference: Anthropic, "Contextual Retrieval" (2024).
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


@dataclass
class Chunk:
    """Minimal chunk representation used by the contextualizer."""
    id: str
    passage_id: str
    title: str
    text: str
    token_count: int = 0
    contextual_text: str = ""

    @property
    def index_text(self) -> str:
        return self.contextual_text or self.text


class LLMClient(Protocol):
    """Minimal interface expected by Contextualizer."""

    def chat(self, system: str, user: str, max_tokens: int = 64) -> str:
        ...


CONTEXTUALIZE_PROMPT = (
    "Here is a document titled '{title}':\n<document>\n{doc}\n</document>\n\n"
    "Here is a chunk from it:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, single-sentence context (<=25 words) that situates this chunk within "
    "the document so it can be retrieved on its own. Answer with the sentence only."
)


class Contextualizer:
    def __init__(self, llm: LLMClient, limit: int = 0, workers: int = 32):
        self.llm = llm
        self.limit = limit
        self.workers = workers
        self.sys_prompt = (
            "You write concise retrieval context. Output one sentence, nothing else."
        )

    @staticmethod
    def estimate(n_chunks: int, sec_per_call: float = 0.15) -> dict:
        """Rough wall-clock estimate for the contextualization pass."""
        return {"chunks": n_chunks, "approx_minutes": round(n_chunks * sec_per_call / 60, 1)}

    def _contextualize_one(self, c: Chunk, doc_lookup: dict[str, str]) -> Chunk:
        """Generate and prepend situating context for a single chunk."""
        doc = doc_lookup.get(c.passage_id, c.text)[:4000]
        user = CONTEXTUALIZE_PROMPT.format(title=c.title, doc=doc, chunk=c.text)
        try:
            ctx = self.llm.chat(self.sys_prompt, user, max_tokens=64).strip().replace("\n", " ")
        except Exception:
            ctx = ""
        c.contextual_text = (ctx + "\n" + c.text) if ctx else c.text
        return c

    def contextualize(
        self, chunks: list[Chunk], doc_lookup: dict[str, str]
    ) -> list[Chunk]:
        """
        Run contextualization over chunks with concurrent LLM calls.

        Args:
            chunks: chunks to contextualize.
            doc_lookup: mapping passage_id -> full document text.

        Returns:
            The same list of chunks with `contextual_text` populated.
        """
        todo = chunks if not self.limit else chunks[: self.limit]
        print(f"[ctx] estimate: {self.estimate(len(todo))} (concurrent x{self.workers})")

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(lambda c: self._contextualize_one(c, doc_lookup), todo))

        # Uncapped remainder keeps raw text.
        for c in chunks[len(todo) :]:
            c.contextual_text = c.text

        return chunks


def contextualize_or_load(
    chunks: list[Chunk],
    llm: LLMClient,
    doc_lookup: dict[str, str],
    checkpoint_path: Path,
    limit: int = 0,
    workers: int = 32,
) -> list[Chunk]:
    """
    Contextualize chunks, or load from a checkpoint if it exists.

    This avoids re-running the expensive LLM pass on every notebook restart.
    """
    if checkpoint_path.exists():
        print(f"[ctx] loading checkpoint {checkpoint_path.name}")
        return pickle.load(open(checkpoint_path, "rb"))

    out = Contextualizer(llm, limit=limit, workers=workers).contextualize(chunks, doc_lookup)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(checkpoint_path, "wb"))
    return out
