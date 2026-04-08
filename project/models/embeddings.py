# project/models/embeddings.py
import os
import hashlib
import logging
import numpy as np
import tiktoken
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".embedding_cache")
MODEL = "text-embedding-3-small"
client = OpenAI()


def format_api_string(api: dict) -> str:
    """Format an API entry into an embedding-ready string.

    Format: "[category] > [tool_name]: [name] — [description] | Parameters: [param1, param2, ...]"
    """
    params = ", ".join(p.get("name", "") for p in api.get("parameters", []))
    return (
        f"{api['category']} > {api['tool_name']}: {api['name']} — "
        f"{api['description']} | Parameters: {params}"
    ).strip()


def get_embeddings(texts: list[str], model: str = MODEL, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Get embeddings for a list of texts. Uses disk cache keyed by text hash.

    Returns array of shape (len(texts), embedding_dim).
    Batches API calls in chunks of 100 to avoid rate limits.
    """
    cache_dir.mkdir(exist_ok=True)
    if not texts:
        return np.empty((0, 1536), dtype=np.float32)
    results = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(texts):
        key = hashlib.md5(f"{model}::{text}".encode()).hexdigest()
        cache_file = cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                results[i] = np.load(cache_file)
                continue  # cache hit, skip to next
            except Exception:
                logger.warning(f"Corrupted cache file {cache_file}, re-fetching")
                cache_file.unlink(missing_ok=True)
        uncached_indices.append(i)
        uncached_texts.append(text)

    if uncached_texts:
        logger.info(f"Fetching {len(uncached_texts)} embeddings from API (cache: {len(texts) - len(uncached_texts)} hits)")

    # Batch API calls in chunks of 100
    # text-embedding-3-small has an 8192 token limit; truncate to 8191 tokens exactly
    enc = tiktoken.encoding_for_model(model)
    MAX_TOKENS = 8191

    def _truncate(text: str) -> str:
        tokens = enc.encode(text)
        return enc.decode(tokens[:MAX_TOKENS]) if len(tokens) > MAX_TOKENS else text

    for chunk_start in range(0, len(uncached_texts), 100):
        chunk = [_truncate(t) for t in uncached_texts[chunk_start:chunk_start + 100]]
        try:
            response = client.embeddings.create(input=chunk, model=model)
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding API call failed for chunk starting at {chunk_start}: {e}") from e
        for j, emb_obj in enumerate(response.data):
            idx = uncached_indices[chunk_start + j]
            vec = np.array(emb_obj.embedding, dtype=np.float32)
            key = hashlib.md5(f"{model}::{uncached_texts[chunk_start + j]}".encode()).hexdigest()
            np.save(cache_dir / f"{key}.npy", vec)
            results[idx] = vec

    return np.stack(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
    from load_toolbench import load_api_corpus

    corpus = load_api_corpus()
    sample_api = corpus[0]

    formatted = format_api_string(sample_api)
    assert isinstance(formatted, str) and len(formatted) > 10
    assert sample_api["category"] in formatted
    assert sample_api["name"] in formatted
    print(f"✓ Format: {formatted[:120]}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping embedding test — set OPENAI_API_KEY to run")
    else:
        embs = get_embeddings(["hello world", "foo bar"], model=MODEL)
        assert embs.shape == (2, 1536), f"Expected (2, 1536), got {embs.shape}"
        print(f"✓ Embedding shape: {embs.shape}")
        embs2 = get_embeddings(["hello world", "foo bar"], model=MODEL)
        assert np.allclose(embs, embs2), "Cache returned different embeddings"
        print("✓ Cache hit works")
