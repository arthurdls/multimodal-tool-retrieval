# project/retrieval/retriever.py
import numpy as np
import faiss


def build_faiss_index(corpus_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from embeddings.

    Normalizes embeddings to unit length so inner product = cosine similarity.
    corpus_embeddings shape: (n_apis, dim)
    """
    embs = corpus_embeddings.astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms == 0, 1, norms)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def retrieve_top_k(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatIP,
    k: int = 10
) -> list[int]:
    """Return indices of top-k most similar APIs for a single query embedding.

    Normalizes the query before search.
    """
    q = query_embedding.astype(np.float32).reshape(1, -1)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    _, indices = index.search(q, k)
    return indices[0].tolist()


if __name__ == "__main__":
    np.random.seed(42)
    n, dim = 1000, 1536
    corpus_embs = np.random.randn(n, dim).astype(np.float32)

    index = build_faiss_index(corpus_embs)
    assert index.ntotal == n, f"Expected {n} vectors, got {index.ntotal}"
    print(f"✓ FAISS index built: {index.ntotal} vectors")

    query = np.random.randn(dim).astype(np.float32)
    top_k = retrieve_top_k(query, index, k=5)

    assert len(top_k) == 5, f"Expected 5 results, got {len(top_k)}"
    assert all(0 <= i < n for i in top_k), "Index out of bounds"
    assert len(set(top_k)) == 5, "Duplicate indices in top-k"
    print(f"✓ Top-5 indices: {top_k}")
    print("✓ All assertions passed")
