import os
import re
import tqdm
import json
import argparse
import time
import sys
sys.path.append(".")
import rag_gym
from src.data_loader import KIQA

def rollout_prm(
    llm_name = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    agent_type = "direct",
    data = "medqa",
    split = "train",
    rerank_llm_name = "OpenAI/gpt-4o",
    retriever_name = None,
    corpus_name = None,
    k = 32,
    rrf_k = 60,
    max_iterations = 10,
    temperature = 1.0,
    n_actions = 10,
    cache_dir = "../huggingface/hub",
    api = False,
    save_dir = "./rollouts",
    n = 1,
    i = 0,
):
    save_dir = os.path.join(save_dir, "prm", data, agent_type, llm_name.replace('/', '_'), f"{max_iterations}")

    if agent_type in ["rag", "react", "search_o1", "research"]:
        retriever_name = retriever_name if retriever_name else "RRF-2" if data == "medqa" else "RRF-BGE"
        corpus_name = corpus_name if corpus_name else "MedText" if data == "medqa" else "Wikipedia_HotpotQA" if data == "hotpotqa" else "Wikipedia"
        if "rrf" in retriever_name.lower():
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}_{rrf_k}"
        else:
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}"
        save_dir = os.path.join(save_dir, retrieval_suffix)
    
    reward_model = None
    rerank_model = None
    if agent_type in ["direct", "cot", "rag"]:
        if data == "medqa":
            reward_model = rag_gym.EMReward()
        else:
            reward_model = rag_gym.F1Reward()
    else:
        rerank_model = rag_gym.LMReranker(rerank_llm_name=rerank_llm_name, api=True if "gpt" in rerank_llm_name.lower() else False)
        save_dir = os.path.join(save_dir, rerank_llm_name.replace('/', '_'))
    os.makedirs(save_dir, exist_ok=True)

    dataset = KIQA(data, split=split)
    curr_range = [j for j in range(len(dataset)) if j % n == i]

    
    outcome_reward_labeler = rag_gym.EMReward() if data == "medqa" else rag_gym.F1Reward()

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
        api = api,
        cache_dir = cache_dir
    )

    env = rag_gym.make(
        retriever_name = retriever_name,
        corpus_name = corpus_name,
        max_iter = max_iterations,
        k = k,
        rrf_k = rrf_k,
        cache = True,
        HNSW = True
    )

    for idx in tqdm.tqdm(curr_range):
        item = dataset[idx]
        index = item.get('_id', item.get('index', item.get('id', str(idx))))
        try:
            if os.path.exists(os.path.join(save_dir, f"{index}.json")) and 'prediction' in json.load(open(os.path.join(save_dir, f"{index}.json"))):
                if "###" not in json.dumps(json.load(open(os.path.join(save_dir, f"{index}.json")))["history"][-1]["content"]["history"]):
                    continue
        except:
            pass

        question = f"{item['question']} {item.get('options', '')}".strip()
        answer = item.get("answer", None)

        history = []
        action_cache = []
        observation, info = env.reset(
            question = question,
            truth = answer
        )
        history.append({"type": "state", "content": observation.return_as_json()})
        if agent_type in ["search_o1", "research"]:
            agent.rag_module.qa_cache = {}
        
        try:
            for _ in range(max_iterations):
                actions = agent.generate_action(
                    state = observation,
                    temperature = temperature,
                    num_actions = n_actions
                )
                if reward_model:
                    rewards = reward_model(observation, actions)
                    assert len(rewards) == len(actions)
                else:
                    if agent_type in ["search_o1", "research"]:
                        rewards = rerank_model(observation, actions, qa_cache = agent.rag_module.qa_cache)
                    else:
                        rewards = rerank_model(observation, actions)
                    assert len(rewards) == len(actions)
                    # raise NotImplementedError
                    # rewards should be the reciprocals


                action_cache.append(
                    {
                        "curr_iter": env.curr_iter,
                        "actions": [
                            {
                                "action": actions[act_idx].return_as_json(),
                                "action_string": actions[act_idx].return_as_string(),
                                "reward": reward
                            } for act_idx, reward in enumerate(rewards)
                        ]
                    }
                )
                action = actions[rewards.index(max(rewards))]

                history.append({"type": "action", "content": action.return_as_json(), "string": action.return_as_string()})
                with open(os.path.join(save_dir, f"{index}.json"), "w") as f:
                    json.dump({**item, "history": history, "action_cache": action_cache}, f, indent=4)
                observation, reward, terminated, truncated, info = env.step(action)
                history.append({"type": "state", "content": observation.return_as_json()})
                with open(os.path.join(save_dir, f"{index}.json"), "w") as f:
                    json.dump({**item, "history": history, "action_cache": action_cache}, f, indent=4)
                if agent_type in ["rag", "react", "search_o1", "research"]:
                    with open(os.path.join(save_dir, f"{index}_qd_cache.json"), "w") as f:
                        json.dump({qd["query"]:qd["documents"] for qd in observation.history.qd_list}, f, indent=4)
                if agent_type in ["search_o1", "research"]:
                    with open(os.path.join(save_dir, f"{index}_qa_cache.json"), "w") as f:
                        json.dump(agent.rag_module.qa_cache, f, indent=4)
                if terminated:
                    break

            item["prediction"] = observation.answer
            item["outcome_reward"] = outcome_reward_labeler(state = observation, actions = action) 
            item["history"] = history
            item["action_cache"] = action_cache

            with open(os.path.join(save_dir, f"{index}.json"), "w") as f:
                json.dump(item, f, indent=4)
        except Exception as E:
            error_class = E.__class__.__name__
            os.makedirs("errors", exist_ok=True)
            with open(os.path.join("errors", f"error_{data}_{split}_{index}.txt"), "w") as f:
                f.write(f"{error_class}: {str(E)}\n\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, default="direct")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--rerank_llm_name", type=str, default="OpenAI/gpt-4o")
    parser.add_argument("--retriever_name", type=str, default=None)
    parser.add_argument("--corpus_name", type=str, default=None)
    parser.add_argument("--data", type=str, default="medqa")
    parser.add_argument("--save_dir", type=str, default="./rollouts")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_actions", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="../huggingface/hub")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--api", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()


    agent_type = args.agent_type
    llm_name = args.llm_name
    rerank_llm_name = args.rerank_llm_name
    retriever_name = args.retriever_name
    corpus_name = args.corpus_name
    data = args.data
    save_dir = args.save_dir
    k = args.k
    rrf_k = args.rrf_k
    max_iterations = args.max_iterations
    temperature = args.temperature
    n_actions = args.n_actions
    cache_dir = args.cache_dir
    api = False if args.api is None else True
    n = args.n
    i = args.i

    rollout_prm(
        llm_name = llm_name,
        agent_type = agent_type,
        data = data,
        split = "train",
        rerank_llm_name = rerank_llm_name,
        retriever_name = retriever_name,
        corpus_name = corpus_name,
        k = k,
        rrf_k = rrf_k,
        max_iterations = max_iterations,
        temperature = temperature,
        n_actions = n_actions,
        cache_dir = cache_dir,
        api = api,
        save_dir = save_dir,
        n = n,
        i = i,
    )