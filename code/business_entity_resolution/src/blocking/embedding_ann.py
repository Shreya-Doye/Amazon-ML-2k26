"""
Multilingual Dense Semantic Embedding & FAISS ANN Blocking Module.
Catches zero-lexical-overlap semantic matches, cross-script pairs, and DBA trade names.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingGenerator:
    """
    Multilingual embedding generator with GPU batching and L2 normalization.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        fallback_model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.batch_size = batch_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load embedding model with automatic fallback if primary model is unavailable."""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for EmbeddingGenerator.")
        try:
            print(f"[EmbeddingGenerator] Loading model: {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"[EmbeddingGenerator] Warning: Failed to load {self.model_name} ({e}). Trying fallback: {self.fallback_model_name}")
            self.model = SentenceTransformer(self.fallback_model_name, device=self.device)

    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode a list of text strings into unit L2-normalized float32 vectors.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


class FAISSANNIndex:
    """
    FAISS Approximate Nearest Neighbor (ANN) Index for fast cosine similarity search.
    """
    def __init__(self, dimension: int, nlist: int = 100):
        self.dimension = dimension
        self.nlist = nlist
        self.index = None
        self.entity_ids: List[str] = []

    def build(self, embeddings: np.ndarray, entity_ids: List[str], use_ivf: bool = False) -> "FAISSANNIndex":
        """
        Build index over embeddings matrix.
        Uses IndexFlatIP for exact cosine or IndexIVFPQ for large collections.
        """
        if faiss is None:
            raise ImportError("faiss is required for FAISSANNIndex.")

        self.entity_ids = list(entity_ids)
        n, d = embeddings.shape

        if use_ivf and n >= max(1000, self.nlist * 4):
            quantizer = faiss.IndexFlatIP(d)
            # 8 sub-quantizers, 8 bits each
            n_sub = min(8, d // 8) if d >= 16 else 1
            index = faiss.IndexIVFPQ(quantizer, d, self.nlist, n_sub, 8)
            index.train(embeddings)
            index.add(embeddings)
            self.index = index
        else:
            # Exact inner product (equivalent to cosine since embeddings are L2 normalized)
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            self.index = index

        return self

    def search(self, query_embeddings: np.ndarray, k: int = 20) -> List[Set[str]]:
        """
        Retrieve top-k nearest candidate entity IDs for each query embedding.
        """
        if self.index is None or len(self.entity_ids) == 0:
            return [set() for _ in range(len(query_embeddings))]

        actual_k = min(k, len(self.entity_ids))
        distances, indices = self.index.search(query_embeddings, actual_k)

        results = []
        for row in indices:
            cand_set = {self.entity_ids[idx] for idx in row if idx >= 0 and idx < len(self.entity_ids)}
            results.append(cand_set)
        return results
