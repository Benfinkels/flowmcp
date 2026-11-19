import math
from typing import Callable, List, Optional, Sequence

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EmbeddingFunction = Callable[[List[str]], Sequence[Sequence[float]]]


def _load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install embedding support with `pip install flowmcp[embeddings]`."
        ) from exc
    return SentenceTransformer(model_name)


class Retriever:
    """Retrieve chunks by cosine similarity between text embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedder: Optional[EmbeddingFunction] = None,
    ):
        self.model_name = model_name
        self._embedder = embedder
        self._model = None
        self.chunks: List[str] = []
        self._embeddings: List[List[float]] = []

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if self._embedder is not None:
            encoded = self._embedder(texts)
        else:
            if self._model is None:
                self._model = _load_embedding_model(self.model_name)
            encoded = self._model.encode(texts, convert_to_numpy=True)

        vectors = [[float(value) for value in vector] for vector in encoded]
        if len(vectors) != len(texts):
            raise ValueError("The embedding model must return one vector per text.")
        if any(not vector for vector in vectors):
            raise ValueError("The embedding model returned an empty vector.")
        if vectors and any(len(vector) != len(vectors[0]) for vector in vectors):
            raise ValueError("The embedding model returned vectors with inconsistent dimensions.")
        return vectors

    def add(self, chunks: List[str]):
        if not chunks:
            return

        embeddings = self._encode(chunks)
        if self._embeddings and any(
            len(vector) != len(self._embeddings[0]) for vector in embeddings
        ):
            raise ValueError("The embedding model returned vectors with inconsistent dimensions.")
        self.chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        left_norm = math.sqrt(math.fsum(value * value for value in left))
        right_norm = math.sqrt(math.fsum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        dot_product = math.fsum(a * b for a, b in zip(left, right))
        return dot_product / (left_norm * right_norm)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self.chunks or top_k <= 0 or not query.strip():
            return []

        query_embedding = self._encode([query])[0]
        if len(query_embedding) != len(self._embeddings[0]):
            raise ValueError("The embedding model returned vectors with inconsistent dimensions.")
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda index: self._cosine_similarity(
                query_embedding, self._embeddings[index]
            ),
            reverse=True,
        )
        return [self.chunks[index] for index in ranked[:top_k]]
