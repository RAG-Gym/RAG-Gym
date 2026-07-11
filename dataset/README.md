# `dataset`

Evaluation splits and process-supervision samples for the four benchmarks used
in RAG-Gym. The multi-hop datasets are scored with Exact Match (EM) and
token-level F1; MedQA is scored with accuracy.

## Benchmarks

| Folder / file | Benchmark | Domain | Evaluation split | Metric |
|---------------|-----------|--------|------------------|--------|
| `hotpotqa/` | HotpotQA | Open-domain (Wikipedia) | Last 1,000 dev questions | EM / F1 |
| `2wikimultihopqa/` | 2WikiMultihopQA | Open-domain (Wikipedia) | Last 1,000 dev questions | EM / F1 |
| `bamboogle/` | Bamboogle | Open-domain (Wikipedia) | Full 125-question test set | EM / F1 |
| `medqa/` | MedQA (USMLE) | Medical | Standard 1,273-question test set | Accuracy |

## Files

| Path | Contents |
|------|----------|
| `hotpotqa/hotpot_dev_fullwiki_v1.json` | HotpotQA full-wiki dev set. |
| `hotpotqa/hotpot_evaluate_v1.py` | Official HotpotQA scorer (also used in `src/`). |
| `2wikimultihopqa/dev.json` | 2WikiMultihopQA dev set, with native question-type labels (comparison, bridge_comparison, compositional, inference). |
| `bamboogle/Bamboogle_Prerelease.tsv` | Bamboogle two-hop questions and answers. |
| `medqa/phrases_no_exclude_test.jsonl` | MedQA English four-option test questions. |
| `hotpotqa_dev_1000.csv` | The 1,000 HotpotQA evaluation questions used in the paper. |
| `medqa_train_1000.csv` | The 1,000 MedQA training questions sampled to build process-reward data. |

## Splits used in the paper

- **Evaluation.** HotpotQA and 2WikiMultihopQA use the last 1,000 questions of
  their development sets; Bamboogle uses its full 125-question test set; MedQA
  uses the standard 1,273-question English test set.
- **Process supervision.** Process-reward trajectories are built from 1,000
  training questions (the first 1,000 for HotpotQA; a 1,000-question sample for
  MedQA), disjoint from the evaluation splits.
- **Generalization.** Out-of-domain results are obtained by evaluating agents
  tuned on HotpotQA directly on 2WikiMultihopQA and Bamboogle.

Datasets are loaded by `src/data_loader.py` and referenced by `--data`
(`hotpotqa`, `2wikimultihopqa`, `bamboogle`, `medqa`) in the inference,
evaluation, and training scripts.
