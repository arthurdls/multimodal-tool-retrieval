# Project: Multimodal Tool Retrieval via Hard-Negative Contrastive Learning

**Course:** Multimodal AI (MAS.S60 / 6.S985), Spring 2026, MIT
**Team:** Michael Serrano, Arthur De Los Santos, Dylan Mazard

## Overview

Given a natural language instruction, identify the correct API from 4,518 candidates in the ToolBench corpus. We treat user queries and structured API schemas as distinct modalities and study how the quality of negative examples affects retrieval performance under both random and structured (DFSDT failure-path) distractors.

## Model Variants

| Variant | Base Model | Training Signal |
|---------|------------|-----------------|
| V1 baseline (frozen) | `text-embedding-3-small` (1536-d) | None |
| V2 bi-encoder + random neg | `all-MiniLM-L6-v2` (384-d) | In-batch negatives |
| V3 bi-encoder + hard neg | `all-MiniLM-L6-v2` (384-d) | DFSDT hard negatives |
| V4 hierarchical loss | `all-MiniLM-L6-v2` (384-d) | Hard neg + category-level InfoNCE (lambda = 0.1) |
| V5 tri-modal | `all-MiniLM-L6-v2` (384-d) | Hard neg + code-signature alignment (lambda_code = 0.5) |

Training: 5 epochs, batch size 64, linear warmup (10%), AdamW (sentence-transformers default LR), 7 explicit hard negatives per example for V3/V4/V5. Single NVIDIA A100. Per-variant training time ranges from about 4 min (V2) to 25 min (V5).

## Results

### Full-Corpus Retrieval

| Variant | R@1 | R@5 | R@10 |
|---------|-----|-----|------|
| V1 baseline | 0.9618 | 0.9726 | 0.9779 |
| V2 random neg | 0.9730 | 0.9851 | 0.9902 |
| V3 hard neg | 0.9723 | **0.9862** | 0.9933 |
| V4 hierarchical | 0.9723 | 0.9848 | **0.9973** |
| V5 tri-modal | 0.9705 | 0.9824 | 0.9862 |

### Restricted-Pool Evaluation (100 candidates)

R@5 by negative condition:

| Variant | Random | Sibling | DFSDT | Random to DFSDT drop |
|---------|--------|---------|-------|----------------------|
| V1 baseline | 0.8444 | 0.4611 | 0.3944 | 53% |
| V2 random neg | 1.0000 | 0.7222 | 0.6889 | 31% |
| V3 hard neg | 1.0000 | 0.8667 | 0.8000 | 20% |
| V4 hierarchical | 1.0000 | 0.8333 | **0.9000** | **10%** |
| V5 tri-modal | 1.0000 | 0.6889 | 0.6889 | 31% |

All four fine-tuned variants achieve statistically significant R@5 gains over the baseline on the DFSDT condition (bootstrap 95% CI, 10,000 resamples). V3 and V4 CIs do not overlap with V2, indicating that explicit hard-negative training is itself significant beyond in-batch negatives.

![Recall@5 Ablation](recall_at_5_ablation.png)

## Repository Structure

```
project/
├── README.md                ← this file
├── requirements.txt         ← Python dependencies
├── data/                    ← data loading and preprocessing
│   ├── load_toolbench.py    ← load API corpus and eval examples
│   └── negative_mining.py   ← random, sibling, and DFSDT negative sampling
├── models/
│   └── embeddings.py        ← OpenAI embedding API with disk cache
├── retrieval/
│   └── retriever.py         ← FAISS index construction and top-k retrieval
├── evaluation/
│   └── metrics.py           ← Recall@k, MRR, batch evaluation
├── notebooks/
│   ├── 01_index_apis.ipynb              ← embed all APIs (run once)
│   ├── 02_baseline_eval.ipynb           ← V1 full-corpus baseline
│   ├── 03_hard_negative_eval.ipynb      ← restricted-pool ablation
│   ├── 04_analysis.ipynb                ← per-category and qualitative analysis
│   ├── 05_train.ipynb                   ← train V2 (random) and V3 (hard neg)
│   ├── 06_eval.ipynb                    ← evaluate V2 and V3
│   ├── 07_train_hierarchical.ipynb      ← train V4 (hierarchical loss)
│   ├── 08_train_trimodal.ipynb          ← train V5 (tri-modal)
│   ├── 09_final_analysis.ipynb          ← bootstrap CIs, t-SNE, final tables
│   └── combined_pipeline_with_results.ipynb  ← end-to-end pipeline
├── results_baseline.json
├── results_hard_negatives.json
└── recall_at_5_ablation.png
```

## Running the Experiments

All experiments run on Google Colab Pro / a single NVIDIA A100. Follow notebooks in order:

1. **[01_index_apis.ipynb](notebooks/01_index_apis.ipynb)**: pre-compute OpenAI embeddings for all API docs. Run once; results are cached.
2. **[02_baseline_eval.ipynb](notebooks/02_baseline_eval.ipynb)**: full-corpus baseline (V1).
3. **[03_hard_negative_eval.ipynb](notebooks/03_hard_negative_eval.ipynb)**: restricted-pool ablation across random, sibling, and DFSDT conditions.
4. **[04_analysis.ipynb](notebooks/04_analysis.ipynb)**: per-category breakdowns and qualitative analysis.
5. **[05_train.ipynb](notebooks/05_train.ipynb)**: train V2 and V3.
6. **[06_eval.ipynb](notebooks/06_eval.ipynb)**: evaluate V2 and V3.
7. **[07_train_hierarchical.ipynb](notebooks/07_train_hierarchical.ipynb)**: train V4 with the category-level loss.
8. **[08_train_trimodal.ipynb](notebooks/08_train_trimodal.ipynb)**: train V5 with typed code signatures.
9. **[09_final_analysis.ipynb](notebooks/09_final_analysis.ipynb)**: bootstrap significance, t-SNE, final tables.

### Requirements

```
pip install -r requirements.txt
```

- OpenAI API key (set as env var `OPENAI_API_KEY` or Colab secret)
- ToolBench dataset (`toolllama_G123_dfs_train.json`, `toolllama_G123_dfs_eval.json`, and `toolenv/tools/`)

## Key Findings

1. **Frozen embeddings are insufficient for hard retrieval conditions.** R@5 drops from 0.84 (random negatives) to 0.39 (DFSDT) at 100-candidate restricted pools, a 53% relative drop.
2. **DFSDT failure traces provide effective training signal.** Hard-negative training (V3) narrows the random-to-DFSDT R@5 drop from 31% (V2) to 20%; adding the category-level term (V4) further reduces it to 10%.
3. **Typed code signatures alone do not improve discrimination.** V5 underperforms V3 and V4 on hard negatives, suggesting that parameter-level signatures lack the functional disambiguation signal needed for confusable APIs.
