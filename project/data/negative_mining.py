import json, random
from pathlib import Path

TOOLBENCH_DIR = Path("toolbench")


def build_random_negatives(api_corpus: list[dict], ground_truth_apis: list[str], n: int = 99) -> list[dict]:
    """Sample n random APIs from the corpus that are not in ground_truth_apis."""
    pool = [a for a in api_corpus if a["action_name"] not in ground_truth_apis]
    return random.sample(pool, min(n, len(pool)))


def build_category_sibling_negatives(
    api_corpus: list[dict],
    ground_truth_apis: list[str],
    api_name_to_entry: dict[str, dict],
    n: int = 99
) -> list[dict]:
    """Sample n APIs from the same category as any ground truth API, excluding ground truth."""
    gt_categories = {
        api_name_to_entry[name]["category"]
        for name in ground_truth_apis
        if name in api_name_to_entry
    }
    pool = [
        a for a in api_corpus
        if a["category"] in gt_categories and a["action_name"] not in ground_truth_apis
    ]
    if len(pool) == 0:
        return build_random_negatives(api_corpus, ground_truth_apis, n)
    return random.sample(pool, min(n, len(pool)))


def build_dfsdt_negatives(
    raw_example: dict,
    api_corpus: list[dict],
    ground_truth_apis: list[str],
    api_name_to_entry: dict[str, dict],
    n: int = 99
) -> list[dict]:
    """Extract APIs from failed/backtracked DFSDT paths in this example's conversation.
    Falls back to category siblings if fewer than n failure-path APIs found."""
    # Extract APIs from failed/backtracked turns
    failure_api_names = []
    for turn in raw_example.get("conversations", []):
        if turn["from"] != "assistant":
            continue
        val = turn["value"]
        is_failure = "give_up_and_restart" in val or (
            "Action:" in val and "Finish" not in val and
            any(f in val for f in ["error", "Error", "failed", "Failed"])
        )
        if is_failure and "Action:" in val:
            try:
                api_name = val.split("Action:")[1].split("\n")[0].strip()
                if api_name and api_name not in ground_truth_apis:
                    failure_api_names.append(api_name)
            except IndexError:
                pass

    # Map names to corpus entries
    failure_apis = [
        api_name_to_entry[name]
        for name in failure_api_names
        if name in api_name_to_entry
    ]

    # Deduplicate
    seen = set()
    unique_failures = []
    for a in failure_apis:
        if a["action_name"] not in seen:
            seen.add(a["action_name"])
            unique_failures.append(a)

    # Fall back to category siblings if not enough
    if len(unique_failures) < n:
        siblings = build_category_sibling_negatives(
            api_corpus, ground_truth_apis + [a["action_name"] for a in unique_failures],
            api_name_to_entry, n - len(unique_failures)
        )
        unique_failures.extend(siblings)

    return unique_failures[:n]


def build_api_lookup(api_corpus: list[dict]) -> dict[str, dict]:
    """Build a dict mapping api name -> api entry for fast lookup."""
    return {a["action_name"]: a for a in api_corpus}


if __name__ == "__main__":
    from load_toolbench import load_api_corpus, load_eval_examples

    corpus = load_api_corpus()
    lookup = build_api_lookup(corpus)
    evals = load_eval_examples()

    sample = evals[0]
    gt = sample["ground_truth_apis"]

    randoms = build_random_negatives(corpus, gt, n=9)
    assert len(randoms) == 9
    assert not any(a["action_name"] in gt for a in randoms)

    siblings = build_category_sibling_negatives(corpus, gt, lookup, n=9)
    assert isinstance(siblings, list) and len(siblings) <= 9 and not any(a["action_name"] in gt for a in siblings)
    assert not any(a["action_name"] in gt for a in siblings)

    with open(TOOLBENCH_DIR / "toolllama_G123_dfs_eval.json") as f:
        raw_evals = json.load(f)

    dfsdt = build_dfsdt_negatives(raw_evals[0], corpus, gt, lookup, n=9)
    assert len(dfsdt) > 0

    print("✓ Random negatives:", len(randoms))
    print("✓ Category sibling negatives:", len(siblings))
    print("✓ DFSDT negatives:", len(dfsdt))
    print("  Sample DFSDT negative:", dfsdt[0]["name"] if dfsdt else "none found")
