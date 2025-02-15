# DPOTrainer
import os
import tqdm
import json
import torch
import argparse
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import sys
sys.path.append(".")
import rag_gym

def dpo_training(
    agent: rag_gym.BaseAgent,
    rollout_dir: str,
    save_dir: str,
    r: int = 256,
    lr: float = 5e-6,
    n_epochs: int = 3,
    batch_size: int = 1,
    accumulation_steps: int = 64,
    eval_ratio: float = 0.1,
    max_prompt_length: int = 4096,
    outcome_reward: str = "em"
):
    os.makedirs(save_dir, exist_ok=True)
    
    reward_labeler = rag_gym.EMReward() if outcome_reward.lower() == "em" else rag_gym.F1Reward()
    # prepare the data

    fnames = [fpath for fpath in sorted(os.listdir(rollout_dir)) if fpath.endswith(".json") and "history" not in fpath and "qa_cache" not in fpath and "qd_cache" not in fpath]

    # preference_data = []
    question_keys = []
    question2instance = {}
    for fname in tqdm.tqdm(fnames):
        try:
            item = json.load(open(os.path.join(rollout_dir, fname)))
            if "prediction" not in item:
                continue
            if os.path.exists(os.path.join(rollout_dir, fname.replace(".json", "_qd_cache.json"))):
                qd_cache = json.load(open(os.path.join(rollout_dir, fname.replace(".json", "_qd_cache.json"))))
            else:
                qd_cache = {}
            if os.path.exists(os.path.join(rollout_dir, fname.replace(".json", "_qa_cache.json"))):
                qa_cache = json.load(open(os.path.join(rollout_dir, fname.replace(".json", "_qa_cache.json"))))
            else:
                qa_cache = {}
        except:
            continue
        if item["history"][-2]["content"]["query"] is not None:
            continue

        question = item["history"][0]["content"]["question"]

        if question not in question2instance:
            question2instance[question] = []
            question_keys.append(question)

        if agent.agent_type == "direct":
            actions = item["action_cache"][0]["actions"]
            neg_actions = [act for act in actions if act["reward"] < 0.5]
            neg_answers = set([act["action"]["answer"] for act in neg_actions])
            if len(neg_actions) == 0:
                continue
            messages = agent.apply_template(rag_gym.State(question = question))
            for ans in neg_answers:
                question2instance[question].append(
                    {
                        "chosen": messages + [{"role": "assistant", "content": f"{{\n\"predicted_answer\": \"{item['answer']}\"\n}}"}],
                        "rejected": messages + [{"role": "assistant", "content": f"{{\n\"predicted_answer\": \"{ans}\"\n}}"}]
                    }
                )
            continue
        # should add a reward value to the final prediction
        if reward_labeler(
            state = rag_gym.State(
                question = "",
                truth = item["answer"]
            ),
            actions = rag_gym.Action(
                answer = item["prediction"]
            )
        ) <= 0.75:
            continue
        for i in range(len(item["action_cache"])):
            actions = item["action_cache"][i]["actions"]
            # actions = [act for act in actions if "Error:" not in act["action_string"]]
            # if len(set([act["reward"] for act in actions])) == 1:
            #     continue
            pos_actions = [act for act in actions if act["reward"] > 0.75]
            neg_actions = [act for act in actions if act["reward"] < 0.5]
            if len(pos_actions) == 0 or len(neg_actions) == 0:
                continue
            state = rag_gym.State(
                question = question,
                history = rag_gym.History(
                    [{"query": query["query"], "documents": qd_cache[query["query"]]} for query in item["history"][i*2]["content"]["history"]]
                ),
                truth = item["answer"]
            )
            if agent_type in ["direct", "cot", "rag", "react"]:
                messages = agent.apply_template(state)
            else:
                messages = agent.apply_template(state, qa_cache=qa_cache)
            pos_actions = set([act["action_string"] for act in pos_actions])
            chosen_messages = [messages + [{"role": "assistant", "content": act}] for act in pos_actions]
            neg_actions = set([act["action_string"] for act in neg_actions])
            rejected_messages = [messages + [{"role": "assistant", "content": act}] for act in neg_actions]

            for cm in chosen_messages:
                for rm in rejected_messages:
                    question2instance[question].append(
                        {
                            "chosen": cm,
                            "rejected": rm
                        }
                    )

    # the dataset should only contain the columns "chosen" and "rejected"
    if eval_ratio == 0:
        eval_keys = []
    else:
        import random
        random.seed(0)
        random.shuffle(question_keys)
        eval_keys = [key for key in question_keys[:round(len(question_keys)*eval_ratio)]]
    train_keys = [key for key in question_keys if key not in eval_keys]
    train_dataset = Dataset.from_list([ins for key in train_keys for ins in question2instance[key]])
    print("Number of training sample:", len(train_dataset))
    if len(eval_keys) == 0:
        eval_dataset = None
        eval_strategy = "no"
    else:
        eval_dataset = Dataset.from_list([ins for key in eval_keys for ins in question2instance[key]])
        eval_strategy = "epoch"
        print("Number of evaluation sample:", len(eval_dataset))

    agent.llm.model.tokenizer.padding_side = 'left'
    agent.llm.model.tokenizer.truncation_side = 'left'
    if agent.llm.model.tokenizer.pad_token is None:
        agent.llm.model.tokenizer.pad_token = agent.llm.model.tokenizer.eos_token
        # agent.llm.model.model.config.pad_token_id = agent.llm.model.tokenizer.eos_token_id
    agent.llm.model.model.model.config.use_cache = False

    # prepare the model
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        lora_alpha=r*2,
        lora_dropout=0.05,
        r=r,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    )
    # model = get_peft_model(reward_model.model, peft_config)

    # prepare the training

    training_args = DPOConfig(
        output_dir = save_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps = accumulation_steps,
        num_train_epochs = n_epochs,
        learning_rate = lr,
        logging_steps = 1,
        eval_strategy = eval_strategy,
        # eval_steps = 50,
        max_prompt_length  = max_prompt_length,
        save_strategy = "epoch",
        save_only_model = True,
    )


    trainer = DPOTrainer(
        model=agent.llm.model.model,
        args=training_args,
        processing_class=agent.llm.model.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # data_collator=collator,
        peft_config=peft_config,
    )

    trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, default="direct")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--rerank_llm_name", type=str, default="OpenAI/gpt-4o")
    parser.add_argument("--retriever_name", type=str, default=None)
    parser.add_argument("--corpus_name", type=str, default=None)
    parser.add_argument("--data", type=str, default="medqa")
    parser.add_argument("--rollout_dir", type=str, default="./rollouts")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="../huggingface/hub")
    parser.add_argument("--r", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=32)
    parser.add_argument("--eval_ratio", type=float, default=0.0)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    args = parser.parse_args()

    agent_type = args.agent_type
    data = args.data
    llm_name = args.llm_name
    rerank_llm_name = args.rerank_llm_name
    retriever_name = args.retriever_name
    corpus_name = args.corpus_name
    k = args.k
    rrf_k = args.rrf_k
    max_iterations = args.max_iterations
    rollout_dir = args.rollout_dir
    cache_dir = args.cache_dir
    r = args.r
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    eval_ratio = args.eval_ratio
    max_prompt_length = args.max_prompt_length

    outcome_reward = "em" if data == "medqa" else "f1"

    type2class = {
        "direct": rag_gym.DirectAgent,
        "cot": rag_gym.CoTAgent,
        "rag": rag_gym.RAGAgent,
        "react": rag_gym.ReActAgent,
        "search_o1": rag_gym.Searcho1Agent,
        "research": rag_gym.ReSearchAgent,
    }

    agent = type2class[agent_type](
        llm_name = llm_name,
        cache_dir = cache_dir,
        api = False,
        model_dtype = torch.bfloat16,
        train_mode = False
    )

    rollout_dir = os.path.join(rollout_dir, "prm", data, agent_type, llm_name.replace('/', '_'), str(max_iterations))

    if agent_type in ["rag", "react", "search_o1", "research"]:
        retriever_name = retriever_name if retriever_name else "RRF-2" if data == "medqa" else "RRF-BGE"
        corpus_name = corpus_name if corpus_name else "MedText" if data == "medqa" else "Wikipedia_HotpotQA" if data == "hotpotqa" else "Wikipedia"
        if "rrf" in retriever_name.lower():
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}_{rrf_k}"
        else:
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}"
        rollout_dir = os.path.join(rollout_dir, retrieval_suffix)
        if agent_type in ["react", "search_o1", "research"]:
            rollout_dir = os.path.join(rollout_dir, rerank_llm_name.replace('/', '_'))

    save_dir = os.path.join(rollout_dir.replace("rollouts", "trained_models"), "dpo", f"{max_prompt_length}_{r}_{lr}_{n_epochs}_{batch_size}_{accumulation_steps}_{eval_ratio}")

    dpo_training(agent, rollout_dir, save_dir, r, lr, n_epochs, batch_size, accumulation_steps, eval_ratio, max_prompt_length, outcome_reward)