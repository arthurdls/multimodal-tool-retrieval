# project/data/load_toolbench.py
import json, os, re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOOLBENCH_DIR = Path("toolbench")  # adjust path as needed in Colab


def load_api_corpus(tools_dir: Path = TOOLBENCH_DIR / "toolenv" / "tools") -> list[dict]:
    """Load all API definitions from the ToolBench tools directory.

    Returns a flat list of API dicts, each with keys:
      name, description, category, tool_name, parameters (list of dicts)
    """
    if not tools_dir.exists():
        raise FileNotFoundError(f"Tools directory not found: {tools_dir.absolute()}")
    apis = []
    for category in sorted(os.listdir(tools_dir)):
        cat_path = tools_dir / category
        if not cat_path.is_dir() or category.startswith("."):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith(".json"):
                continue
            with open(cat_path / fname) as f:
                content = f.read()
            if not content.strip():
                continue
            tool = json.loads(content)
            tool_stem = fname.replace(".json", "")
            tool_name = tool.get("tool_name", tool_stem)
            for endpoint in tool.get("api_list", []):
                ep_name = endpoint.get("name", "")
                snake = re.sub(r"[^a-z0-9]+", "_", ep_name.lower()).strip("_")
                action_name = f"{snake}_for_{tool_stem}"
                apis.append({
                    "name": ep_name,
                    "action_name": action_name,
                    "description": endpoint.get("description", ""),
                    "category": category,
                    "tool_name": tool_name,
                    "parameters": endpoint.get("required_parameters", []) + endpoint.get("optional_parameters", []),
                })
    return apis



def load_eval_examples(eval_path: Path = TOOLBENCH_DIR / "toolllama_G123_dfs_eval.json") -> list[dict]:
    """Load evaluation examples. Each example is a dict with keys:
      id, user_query (first user turn only), ground_truth_apis (list of str, the API names called)
    """
    with open(eval_path) as f:
        raw = json.load(f)

    examples = []
    for idx, ex in enumerate(raw):
        user_query = ""
        ground_truth_apis = []
        for turn in ex.get("conversations", []):
            if turn["from"] == "user" and not user_query:
                user_query = turn["value"]
            elif turn["from"] == "assistant":
                val = turn["value"]
                if "Action:" in val and "give_up_and_restart" not in val:
                    try:
                        api_name = val.split("Action:")[1].split("\n")[0].strip()
                        if api_name and api_name not in ground_truth_apis:
                            ground_truth_apis.append(api_name)
                    except IndexError:
                        logger.warning(f"Failed to extract API name from turn: {val[:100]}")
        if user_query and ground_truth_apis:
            examples.append({
                "id": ex.get("id", ""),
                "user_query": user_query,
                "ground_truth_apis": ground_truth_apis,
                "raw_idx": idx,
            })
    return examples


if __name__ == "__main__":
    corpus = load_api_corpus()
    assert len(corpus) > 10000, f"Expected >10k APIs, got {len(corpus)}"
    assert all("name" in a and "description" in a and "category" in a for a in corpus[:10])

    evals = load_eval_examples()
    assert len(evals) > 0, "No eval examples loaded"
    assert all("user_query" in e and "ground_truth_apis" in e for e in evals[:5])
    print(f"✓ API corpus: {len(corpus)} APIs")
    print(f"✓ Eval examples: {len(evals)}")
    print(f"  Sample API: {corpus[0]}")
    print(f"  Sample query: {evals[0]['user_query'][:100]}")
