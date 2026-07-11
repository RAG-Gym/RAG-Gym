# `rag_gym/algorithms/prm`

Process-supervision training pipeline. The four stages turn agent rollouts into
process-reward data and use it to train the actor (SFT, DPO) and the critic
(process reward model, PRM). All hyperparameter defaults below match the values
reported in Supplementary Table 6 of the paper.

## Pipeline overview

```
rollout.py  ──►  sft/sft.py        (imitate preferred actions)
            ├─►  dpo/dpo.py        (contrast preferred vs. dispreferred)
            └─►  reward/reward.py  (train the process reward model / critic)
```

1. **Trajectory rollout** samples multiple candidate actions per step from an
   untuned agent, keeps successful trajectories, and has an external annotator
   (GPT-4o by default) rank the candidates to form `(state, a+, a-)` preference
   tuples.
2. **SFT / DPO** fine-tune the actor on these preferences.
3. **PRM** trains a Bradley-Terry critic that scores candidate actions at
   inference for critic-guided (Best-of-N) selection.

## Files

| Path | Stage | Description |
|------|-------|-------------|
| `rollout.py` | Data collection | Samples trajectories, retains ground-truth-correct paths, and ranks step candidates with `--rerank_llm_name` (default `OpenAI/gpt-4o`). Writes to `--save_dir` (default `./rollouts`). |
| `sft/sft.py` | Actor (SFT) | LoRA supervised fine-tuning on preferred actions. |
| `dpo/dpo.py` | Actor (DPO) | LoRA direct preference optimization over preference pairs. |
| `reward/reward.py` | Critic (PRM) | Trains the LoRA sequence-classification reward model used as the critic. |

## Quick start

The commands below reproduce the HotpotQA Re$^2$Search pipeline with the
Llama-3.1-8B backbone. Use `--agent_type` to select the architecture
(`direct`, `cot`, `rag`, `react`, `search_o1`, `research`) and `--data` to
select the dataset (`hotpotqa`, `medqa`, ...).

```bash
# 1. Collect process-reward data (trajectory sampling + GPT-4o ranking)
python rag_gym/algorithms/prm/rollout.py \
    --data hotpotqa --agent_type research \
    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --rerank_llm_name OpenAI/gpt-4o

# 2a. Supervised fine-tuning (LoRA)
python rag_gym/algorithms/prm/sft/sft.py \
    --data hotpotqa --agent_type research \
    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct

# 2b. Direct preference optimization (LoRA)
python rag_gym/algorithms/prm/dpo/dpo.py \
    --data hotpotqa --agent_type research \
    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct

# 3. Process reward model / critic (LoRA sequence classification)
python rag_gym/algorithms/prm/reward/reward.py \
    --data hotpotqa --agent_type research \
    --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Key hyperparameters (defaults)

| Stage | Learning rate | LoRA rank `r` | Effective batch | Max length |
|-------|--------------:|--------------:|----------------:|-----------:|
| DPO (`dpo.py`) | `2e-6` | 256 | 32 (accumulation) | 4096 prompt |
| PRM (`reward.py`) | `1e-5` | 256 | 2 (accumulation) | 4096 |

SFT, DPO, and the PRM critic use LoRA adaptation via TRL; PPO (in the paper,
run with OpenRLHF) uses full-parameter tuning. Rollout and training scripts
share `--data`, `--agent_type`, `--retriever_name`, `--corpus_name`, `--k`, and
`--rrf_k` so the environment matches across stages. See Supplementary Table 6
for the complete hyperparameter list.
