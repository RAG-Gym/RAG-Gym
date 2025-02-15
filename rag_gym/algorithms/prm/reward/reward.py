# RewardTrainer
import os
import tqdm
import json
import torch
import argparse
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
# from trl.trainer import reward_trainer
from trl import RewardTrainer, RewardConfig
from trl.trainer.utils import compute_accuracy, RewardDataCollatorWithPadding
from trl.data_utils import maybe_apply_chat_template
from accelerate import PartialState
from transformers.utils import is_peft_available
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
import inspect
import warnings
from dataclasses import FrozenInstanceError, replace
# from typing import Any, Callable, Optional, Union

def custom_tokenize(batch: dict, tokenizer, max_length) -> dict:
    """Tokenize a batch from a reward modelling dataset."""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(batch["chosen"], batch["rejected"]):
        tokenized_chosen = tokenizer(chosen, add_special_tokens=False, truncation=True, max_length=max_length, padding="longest")
        tokenized_rejected = tokenizer(rejected, add_special_tokens=False, truncation=True, max_length=max_length, padding="longest")
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

class CustomRewardTrainer(RewardTrainer):
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None,None,),
        preprocess_logits_for_metrics = None,
        max_length = None,
        peft_config = None,
    ):
        if max_length is not None and args.max_length is not None:
            raise ValueError(
                "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
            )
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`.",
                            UserWarning,
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                model = get_peft_model(model, peft_config)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if processing_class is None:
                raise ValueError(
                    "A processing_class must be specified when using the default RewardDataCollatorWithPadding"
                )
            if max_length is None:
                max_length = 512 if args.max_length is None else args.max_length

            data_collator = RewardDataCollatorWithPadding(processing_class)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Reward, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "input_ids_chosen" and "input_ids_rejected". As a result,
        # the trainer issues the warning: "Could not estimate the number of tokens of the input, floating-point
        # operations will not be computed." To suppress this warning, we set the "estimate_tokens" key in the model's
        # "warnings_issued" dictionary to True. This acts as a flag to indicate that the warning has already been
        # issued.
        model.warnings_issued["estimate_tokens"] = True

        if "input_ids_chosen" not in train_dataset.column_names:
            with PartialState().local_main_process_first():
                fn_kwargs = {"tokenizer": processing_class, "max_length": max_length}
                train_dataset = train_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class})
                train_dataset = train_dataset.map(
                    custom_tokenize,
                    batched=True,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                )

                train_dataset = train_dataset.filter(
                    lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length,
                    num_proc=args.dataset_num_proc,
                )
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(
                        maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class}
                    )
                    eval_dataset = eval_dataset.map(
                        custom_tokenize,
                        fn_kwargs=fn_kwargs,
                        batched=True,
                        num_proc=args.dataset_num_proc,
                    )
                    
                    eval_dataset = eval_dataset.filter(
                        lambda x: len(x["input_ids_chosen"]) <= max_length
                        and len(x["input_ids_rejected"]) <= max_length,
                        num_proc=args.dataset_num_proc,
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

# reward_trainer._tokenize = custom_tokenize
import sys
sys.path.append(".")
import rag_gym

def reward_training(
    agent: rag_gym.BaseAgent,
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    rollout_dir: str,
    save_dir: str,
    r: int = 256,
    alpha: int = 512,
    lr: float = 5e-6,
    n_epochs: int = 3,
    batch_size: int = 1,
    accumulation_steps: int = 64,
    eval_ratio: float = 0.1,
    max_length  = 4096,
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

    reward_tokenizer.padding_side = 'left'
    reward_tokenizer.truncation_side = 'left'
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        # reward_tokenizer.pad_token = "<|end_of_text|>" # 128001
    if reward_model.config.pad_token_id is None:
        # reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        reward_model.config.pad_token_id = -1

    # prepare the model
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        lora_alpha=alpha,
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

    training_args = RewardConfig(
        output_dir = save_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps = accumulation_steps,
        num_train_epochs = n_epochs,
        learning_rate = lr,
        logging_steps = 1,
        eval_strategy = eval_strategy,
        # eval_strategy = "steps",
        # eval_steps = 1,
        max_length = max_length,
        save_strategy = "epoch",
        save_only_model = True,
        center_rewards_coefficient = 0.01
    )

    trainer = CustomRewardTrainer(
        model=reward_model,
        processing_class=reward_tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, default="direct")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--rerank_llm_name", type=str, default="OpenAI/gpt-4o")
    parser.add_argument("--reward_llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--retriever_name", type=str, default=None)
    parser.add_argument("--corpus_name", type=str, default=None)
    parser.add_argument("--data", type=str, default="medqa")
    parser.add_argument("--rollout_dir", type=str, default="./rollouts")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="../huggingface/hub")
    parser.add_argument("--r", type=int, default=256)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    args = parser.parse_args()

    agent_type = args.agent_type
    data = args.data
    llm_name = args.llm_name
    rerank_llm_name = args.rerank_llm_name
    reward_llm_name = args.reward_llm_name
    retriever_name = args.retriever_name
    corpus_name = args.corpus_name
    k = args.k
    rrf_k = args.rrf_k
    max_iterations = args.max_iterations
    rollout_dir = args.rollout_dir
    cache_dir = args.cache_dir
    r = args.r
    alpha = args.alpha if args.alpha else 2 * r
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    max_length = args.max_length
    eval_ratio = args.eval_ratio

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
        train_mode = True
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_llm_name, 
        device_map = "auto", 
        cache_dir = cache_dir, 
        torch_dtype = torch.bfloat16,
        num_labels = 1
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_llm_name, cache_dir=cache_dir)
    
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

    save_dir = os.path.join(rollout_dir.replace("rollouts", "trained_models"), "reward_model", reward_llm_name.replace('/', '_'), f"{r}_{alpha}_{lr}_{n_epochs}_{batch_size}_{accumulation_steps}_{eval_ratio}")
    
    reward_training(agent, reward_model, reward_tokenizer, rollout_dir, save_dir, r, alpha, lr, n_epochs, batch_size, accumulation_steps, eval_ratio, max_length, outcome_reward)