# `src`

Command-line entry points for running agents over a dataset and scoring their
predictions. These wrap the `rag_gym` package so full evaluations can be run
without writing code.

## Files

| File | Description |
|------|-------------|
| `run_pipeline.py` | Runs an agent over a dataset and writes per-question predictions. Supports zero-shot (`--n_actions 1`) and critic-guided Best-of-N (`--n_actions > 1` with `--reward_llm_name`) inference. |
| `evaluate.py` | Runs inference and reports metrics (EM / F1 for the multi-hop datasets, accuracy for MedQA). |
| `data_loader.py` | Loads the evaluation splits from `dataset/` (see `dataset/README.md`). |
| `hotpot_evaluate_v1.py` | Official HotpotQA scoring functions (normalized EM / token-level F1) reused for consistent evaluation. |

## Zero-shot inference

```bash
python src/run_pipeline.py \
    --agent_type research \
    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --retriever_name RRF-BGE --corpus_name Wikipedia \
    --data hotpotqa \
    --n_actions 1 --temperature 0.0 \
    --save_dir ./predictions
```

## Critic-guided (Best-of-N) inference

Provide a trained process reward model with `--reward_llm_name` and sample
multiple candidates per step; the critic selects the highest-scoring action.

```bash
python src/run_pipeline.py \
    --agent_type research \
    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --reward_llm_name RAG-Gym/ReSearch-HotpotQA-PRM \
    --retriever_name RRF-BGE --corpus_name Wikipedia \
    --data hotpotqa \
    --n_actions 10 --temperature 1.0 \
    --save_dir ./predictions
```

## Evaluation

```bash
python src/evaluate.py \
    --agent_type research \
    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --reward_llm_name RAG-Gym/ReSearch-HotpotQA-PRM \
    --retriever_name RRF-BGE --corpus_name Wikipedia \
    --data hotpotqa --n_actions 10 --temperature 1.0
```

## Common arguments

| Argument | Meaning |
|----------|---------|
| `--agent_type` | `direct`, `cot`, `rag`, `react`, `search_o1`, or `research` (Re$^2$Search). |
| `--llm_name` | Actor backbone (local or, with `--api`, an API model). |
| `--reward_llm_name` | PRM checkpoint for critic-guided inference; omit for zero-shot. |
| `--retriever_name` / `--corpus_name` | Retrieval configuration (see `rag_gym/envs/README.md`). |
| `--k` / `--rrf_k` | Documents per query / RRF constant (defaults 32 / 60). |
| `--max_iterations` | Maximum search steps per question (default 10). |
| `--n_actions` | Candidates per step; `1` is zero-shot, `>1` enables Best-of-N. |
| `--temperature` | Sampling temperature (`0.0` for zero-shot, `1.0` for Best-of-N). |
| `--data` | Dataset name matching a folder in `dataset/`. |
| `--save_dir` | Output directory for predictions (default `./predictions`). |
