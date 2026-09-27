"""Embedding models behind one interface: an OpenAI-compatible endpoint, or a local sentence-transformers model."""
from __future__ import annotations

import gc
import os
from typing import Protocol

import numpy as np

from .cache import DiskCache, make_key
from .llm import LLM


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        """Unit-length vectors, one row per text. `kind` is "query" or "document"."""
        ...


def normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def embed_prefix(kind: str) -> str:
    """Some embedding models were trained with task prefixes and retrieve worse without them.
    Set EMBED_DOC_PREFIX / EMBED_QUERY_PREFIX if yours is one of them."""
    return os.environ.get("EMBED_QUERY_PREFIX" if kind == "query" else "EMBED_DOC_PREFIX", "")


class ApiEmbedder:
    """Any OpenAI-compatible /embeddings endpoint (EMBED_MODEL). Every vector is cached on disk,
    so an interrupted run resumes and a re-run costs nothing."""

    def __init__(self, llm: LLM | None = None, model: str | None = None, batch_size: int = 32):
        self.llm = llm or LLM()
        self.model = model or self.llm.embed_model
        if not self.model:
            raise ValueError("set EMBED_MODEL")
        self.name = "api"
        self.batch_size = batch_size
        self.cache = DiskCache("embed-" + self.model.replace(":", "_").replace("/", "_"))

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        prefix = embed_prefix(kind)
        out: list[np.ndarray | None] = [None] * len(texts)
        todo = []
        for i, text in enumerate(texts):
            key = make_key(self.model, prefix + text)
            hit = self.cache.get_array(key)
            if hit is None:
                todo.append((i, key, prefix + text))
            else:
                out[i] = hit
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            for (i, key, _), vec in zip(batch, self.llm.embed([t for _, _, t in batch], model=self.model)):
                arr = np.asarray(vec, dtype=np.float32)
                self.cache.set_array(key, arr)
                out[i] = arr
        return normalize(np.vstack(out)) if out else np.zeros((0, 0), dtype=np.float32)


# short name -> (Hugging Face id, query prompt, document prompt). The prompts are what each model was trained with.
LOCAL_MODELS: dict[str, tuple[str, str, str]] = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-small-multi": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
}


def torch_device() -> str:
    import torch

    if os.environ.get("RETRIEVAL_LAB_DEVICE"):
        return os.environ["RETRIEVAL_LAB_DEVICE"]
    return "mps" if torch.backends.mps.is_available() else "cpu"


def free_torch_memory() -> None:
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class LocalEmbedder:
    """A sentence-transformers model on Apple Metal (or CPU). Loaded on first use; call `close()` to give the memory back."""

    def __init__(self, short_name: str, batch_size: int = 32, max_seq_length: int = 512):
        self.hf_id, self.query_prompt, self.doc_prompt = LOCAL_MODELS[short_name]
        self.name = short_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = None
        self.query_cache = DiskCache(f"st-{short_name}")

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.hf_id, device=torch_device())
            self._model.max_seq_length = min(self.max_seq_length, self._model.max_seq_length or self.max_seq_length)
        return self._model

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        prompt = self.query_prompt if kind == "query" else self.doc_prompt
        if kind != "query":
            return self._encode([prompt + t for t in texts], show_progress=len(texts) > 1000)
        # queries are few and repeat across experiments, so cache them one by one
        out: list[np.ndarray | None] = []
        todo = []
        for i, t in enumerate(texts):
            hit = self.query_cache.get_array(make_key(self.hf_id, prompt + t))
            out.append(hit)
            if hit is None:
                todo.append(i)
        if todo:
            vecs = self._encode([prompt + texts[i] for i in todo])
            for i, v in zip(todo, vecs):
                self.query_cache.set_array(make_key(self.hf_id, prompt + texts[i]), v)
                out[i] = v
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vecs = self._load().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=show_progress
        )
        return np.asarray(vecs, dtype=np.float32)

    def close(self) -> None:
        self._model = None
        free_torch_memory()


def make_embedder(name: str) -> Embedder:
    if name == "api":
        return ApiEmbedder()
    if name in LOCAL_MODELS:
        return LocalEmbedder(name)
    raise ValueError(f"unknown embedder {name!r}; choose api or one of {', '.join(LOCAL_MODELS)}")
