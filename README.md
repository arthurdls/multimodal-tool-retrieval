# Multimodal Tool Retrieval via Hard-Negative Mining

**Course:** Multimodal AI (MAS.S60 / 6.S985) · Spring 2026 · MIT

**Team:** Michael Serrano, Arthur De Los Santos, Dylan Mazard

## Abstract

Agentic systems that interact with large collections of APIs face a fundamental retrieval challenge: given a natural language instruction, identify the correct tool from tens of thousands of candidates. We present a multimodal approach to API retrieval using the ToolBench dataset, treating user queries and structured API schemas as distinct modalities. We establish a dense retrieval baseline using OpenAI `text-embedding-3-small` embeddings over 16,000+ APIs and evaluate three negative-sampling strategies — random, category-sibling, and DFSDT failure-path negatives — to characterize retrieval difficulty under increasingly hard conditions.

## Motivation

The semantic gap between user instructions (continuous, intent-driven natural language) and API documentation (discrete, schema-defined contracts) makes retrieval non-trivial. Distinguishing semantically similar but functionally different APIs — such as `get_current_weather` vs. `get_weather_forecast` — demands representations that capture functional semantics beyond vocabulary overlap.

## Approach

### Dense Retrieval Baseline
All APIs and queries are embedded with OpenAI `text-embedding-3-small`. API embeddings are pre-computed and cached. Retrieval uses a FAISS `IndexFlatIP` (cosine similarity via normalized inner product) over the full 16,000+ API corpus.

### Hard-Negative Mining via DFSDT Traces
We evaluate three negative-sampling strategies of increasing difficulty:

| Strategy | Source | Difficulty |
|----------|--------|------------|
| **Random** | Uniform sample from corpus | Easy — no structural relationship to query |
| **Category Siblings** | Same ToolBench category as ground-truth API | Medium — shared semantic domain |
| **DFSDT Failure Paths** | APIs considered and rejected during successful DFSDT trajectories | Hard — actively confusable candidates |

### Planned Extensions
1. **Hierarchical Contrastive Loss** — Multi-granularity alignment using ToolBench's API → Tool → Category hierarchy, with category-level alignment acting as a regularizer for tail APIs.
2. **Code-Enhanced Tri-Encoder** — BERT encoders for instructions and API docs, plus a CodeBERT encoder for functional code snippets, to disambiguate textually similar but functionally different APIs.

## Results (Midterm)

### Full-Corpus Baseline

| Model | R@1 | R@5 | R@10 | MRR |
|-------|-----|-----|------|-----|
| `text-embedding-3-small` | 0.223 | 0.421 | 0.506 | 0.435 |

### Hard-Negative Ablation (100-candidate restricted pool)

| Negative Condition | R@1 | R@5 | R@10 | MRR |
|--------------------|-----|-----|------|-----|
| Random | 0.520 | 0.914 | 0.951 | 0.926 |
| Category Siblings | 0.417 | 0.772 | 0.839 | 0.808 |
| DFSDT Failure Paths | 0.414 | 0.774 | 0.851 | 0.810 |

Category-level and DFSDT negatives reduce R@1 by ~10 points vs. random, confirming that category semantics represent a meaningful difficulty barrier for the current embedding model.

## Metrics

- **Recall@k** (k = 1, 5, 10) — primary retrieval metric
- **Mean Reciprocal Rank (MRR)**
- **API Selection F1**

## Repository Structure

```
multimodal-tool-retrieval/
├── README.md                ← this file
├── .gitignore
├── .gitmodules
├── Arthur-MMAI-SP26/        ← Arthur's individual repo (submodule)
├── Dylan-MMAI-SP26/         ← Dylan's individual repo (submodule)
└── Michael-MMAI-SP26/       ← Michael's individual repo (submodule)
```

Each team member's submodule contains their homework assignments and a `project/` directory with shared experiment code (data loaders, models, evaluation, notebooks).

## Running the Experiments

All experiments run on Google Colab Pro. Follow the notebooks (in any team member's `project/notebooks/`) in order:

1. **`01_index_apis.ipynb`** — Pre-compute OpenAI embeddings for all API docs. Run once; results are cached.
2. **`02_baseline_eval.ipynb`** — Full-corpus baseline retrieval evaluation (Recall@k, MRR).
3. **`03_hard_negative_eval.ipynb`** — Hard-negative ablation across random, category-sibling, and DFSDT failure-path conditions.

### Requirements

- Google Colab Pro (or any GPU/CPU runtime)
- OpenAI API key (set as Colab secret `OPENAI_API_KEY`)
- ToolBench dataset (`toolllama_G123_dfs_eval.json` + `toolenv/tools/`)

## Team Repositories

| Member | Repository |
|--------|-----------|
| Arthur De Los Santos | [MMAI-SP26](https://github.com/arthurdls/MMAI-SP26) |
| Dylan Mazard | [MMAI-Spring2026](https://github.com/Dmazard/MMAI-Spring2026) |
| Michael Serrano | [mmai](https://github.com/michaelyserrano/mmai) |

## References

- Qin et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs" (2024)
- Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
- Patil et al. "Gorilla: Large Language Model Connected with Massive APIs" (2023)
- Feng et al. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages" (2020)
- Khosla et al. "Supervised Contrastive Learning" (2020)
- Karpukhin et al. "Dense Passage Retrieval for Open-Domain Question Answering" (2020)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
