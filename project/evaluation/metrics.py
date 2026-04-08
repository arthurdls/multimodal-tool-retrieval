# project/evaluation/metrics.py


def recall_at_k(retrieved_indices: list[int], ground_truth_indices: list[int], k: int) -> float:
    """Fraction of ground truth APIs found in the top-k retrieved indices.

    Returns 1.0 if ground_truth_indices is empty (vacuously true).
    """
    if not ground_truth_indices:
        return 1.0
    top_k = set(retrieved_indices[:k])
    hits = sum(1 for g in ground_truth_indices if g in top_k)
    return hits / len(ground_truth_indices)


def mean_reciprocal_rank(retrieved_indices: list[int], ground_truth_indices: list[int]) -> float:
    """1/rank of the first ground truth API in the retrieved list.

    Returns 0.0 if no ground truth API appears in the retrieved list.
    """
    gt_set = set(ground_truth_indices)
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in gt_set:
            return 1.0 / rank
    return 0.0


def evaluate_batch(
    all_retrieved: list[list[int]],
    all_ground_truths: list[list[int]],
    ks: list[int] = [1, 5, 10]
) -> dict[str, float]:
    """Run Recall@k for each k and MRR over a batch.

    Returns a dict mapping metric name -> average score:
      {"recall@1": ..., "recall@5": ..., "recall@10": ..., "mrr": ...}
    """
    results = {f"recall@{k}": 0.0 for k in ks}
    results["mrr"] = 0.0
    n = len(all_retrieved)
    if n == 0:
        return results
    for retrieved, gt in zip(all_retrieved, all_ground_truths):
        for k in ks:
            results[f"recall@{k}"] += recall_at_k(retrieved, gt, k)
        results["mrr"] += mean_reciprocal_rank(retrieved, gt)
    return {key: val / n for key, val in results.items()}


if __name__ == "__main__":
    # Known-answer tests
    retrieved = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    gt = [2, 6]

    assert recall_at_k(retrieved, gt, k=1) == 0.0, "No GT in top-1"
    assert recall_at_k(retrieved, gt, k=3) == 0.5, "One of 2 GT in top-3"
    assert recall_at_k(retrieved, gt, k=5) == 1.0, "Both GT in top-5"
    print("✓ recall_at_k")

    assert mean_reciprocal_rank(retrieved, gt) == 1/2, "First GT (idx=2) at rank 2"
    assert mean_reciprocal_rank([5, 9, 3], [0]) == 0.0, "GT not found -> 0"
    print("✓ mean_reciprocal_rank")

    results = evaluate_batch(
        [[0, 2, 4], [1, 3, 5]],
        [[2], [3]],
        ks=[1, 3]
    )
    assert "recall@1" in results and "recall@3" in results and "mrr" in results
    assert results["recall@3"] == 1.0, f"Expected 1.0, got {results['recall@3']}"
    assert results["recall@1"] == 0.0, f"Expected 0.0, got {results['recall@1']}"
    print("✓ evaluate_batch:", results)

    # Edge cases
    assert recall_at_k([], [], k=5) == 1.0, "Empty GT -> 1.0"
    assert evaluate_batch([], [], ks=[1, 5]) == {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0}
    print("✓ Edge cases pass")
