import os
import re
import time
import tqdm
import json
import argparse
from liquid import Template
import sys
sys.path.append(".")
import rag_gym
from src.data_loader import KIQA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, default="direct")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--rag_llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--reward_llm_name", type=str, default=None)
    parser.add_argument("--retriever_name", type=str, default=None)
    parser.add_argument("--corpus_name", type=str, default=None)
    parser.add_argument("--data", type=str, default="medqa")
    parser.add_argument("--save_dir", type=str, default="./predictions")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--n_actions", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cache_dir", type=str, default="../huggingface/hub")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--api", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    agent_type = args.agent_type
    llm_name = args.llm_name
    rag_llm_name = args.rag_llm_name
    reward_llm_name = args.reward_llm_name
    retriever_name = args.retriever_name
    corpus_name = args.corpus_name
    data = args.data
    save_dir = args.save_dir
    k = args.k
    rrf_k = args.rrf_k
    max_iterations = args.max_iterations
    n_actions = args.n_actions if reward_llm_name else 1
    temperature = args.temperature if reward_llm_name else 0.0
    cache_dir = args.cache_dir
    api = False if args.api is None else True
    
    save_dir = os.path.join(save_dir, data, agent_type, '_'.join([name.replace('/', '_') for name in [llm_name, reward_llm_name] if name is not None]), f"{max_iterations}" if n_actions == 1 else f"{max_iterations}_{n_actions}_{temperature}")

    dataset = KIQA(data)
    curr_range = [j for j in range(len(dataset)) if j % args.n == args.i]

    if agent_type in ["rag", "react", "search_o1", "research"]:
        retriever_name = retriever_name if retriever_name else "RRF-2" if data == "medqa" else "RRF-BGE"
        corpus_name = corpus_name if corpus_name else "MedText" if data == "medqa" else "Wikipedia_HotpotQA" if data == "hotpotqa" else "Wikipedia"
        if "rrf" in retriever_name.lower():
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}_{rrf_k}"
        else:
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}"
        save_dir = os.path.join(save_dir, retrieval_suffix)
    os.makedirs(save_dir, exist_ok=True)

    type2class = {
        "direct": rag_gym.DirectAgent,
        "cot": rag_gym.CoTAgent,
        "rag": rag_gym.RAGAgent,
        "react": rag_gym.ReActAgent,
        "search_o1": rag_gym.Searcho1Agent,
        "research": rag_gym.ReSearchAgent,
    }
    kwargs = {}
    if agent_type in ["search_o1", "research"]:
        kwargs = {"rag_llm_name": rag_llm_name}

    agent = type2class[agent_type](
        llm_name = llm_name,
        api = api,
        cache_dir = cache_dir,
        reward_llm_name = reward_llm_name,
        **kwargs
    )

    max_iterations = max_iterations - 1 if agent_type in ["react", "search_o1"] else max_iterations

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
            existing_pred = json.load(open(os.path.join(save_dir, f"{index}.json")))['prediction']
            if "Error:" not in open(os.path.join(save_dir, f"{index}.json")).read():
                if len(json.load(open(os.path.join(save_dir, f"{index}.json")))["history"]) > 1:
                    continue
        except:
            pass
        question = f"{item['question']} {item.get('options', '')}".strip()
        answer = item.get("answer", None)

        history = []
        action_cache = []
        observation, info = env.reset(
            question = question,
        )
        history.append({"type": "state", "content": observation.return_as_json()})
        if agent_type in ["search_o1", "research"]:
            agent.rag_module.qa_cache = {}

        for _ in range(max_iterations):
            
            if n_actions == 1 or (agent_type == "rag" and env.curr_iter == 0):
                action = agent.generate_action(
                    state = observation,
                    temperature = 0.0,
                )[0]
            else:
                actions = agent.generate_action(
                    state = observation,
                    temperature = temperature,
                    num_actions = n_actions
                )
                if agent_type in ["search_o1", "research"]:
                    rewards = agent.score(observation, actions, qa_cache=agent.rag_module.qa_cache)
                else:
                    rewards = agent.score(observation, actions)
                assert len(rewards) == len(actions)
                action_cache.append(
                    {
                        "curr_iter": env.curr_iter,
                        "actions": [
                            {
                                "content": actions[act_idx].return_as_json(),
                                "string": actions[act_idx].return_as_string(),
                                "reward": reward
                            } for act_idx, reward in enumerate(rewards)
                        ]
                    }
                )
                action = actions[rewards.index(max(rewards))]

            history.append({"type": "action", "content": action.return_as_json(), "string": action.return_as_string()})
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
            if truncated:
                if agent_type in ["react", "search_o1"]:
                    if n_actions == 1:
                        action = agent.generate_action(
                            state = observation,
                            temperature = 0.0,
                        )[0]
                    else:
                        actions = agent.generate_action(
                            state = observation,
                            temperature = temperature,
                            num_actions = n_actions
                        )
                        if agent_type == "search_o1":
                            rewards = agent.score(observation, actions, qa_cache=agent.rag_module.qa_cache)
                        else:
                            rewards = agent.score(observation, actions)
                        assert len(rewards) == len(actions)
                        action_cache.append(
                            {
                                "curr_iter": env.curr_iter,
                                "actions": [
                                    {
                                        "content": actions[act_idx].return_as_json(),
                                        "string": actions[act_idx].return_as_string(),
                                        "reward": reward
                                    } for act_idx, reward in enumerate(rewards)
                                ]
                            }
                        )
                        action = actions[rewards.index(max(rewards))]
                    history.append({"type": "action", "content": action.return_as_json(), "string": action.return_as_string()})
                    observation, reward, terminated, truncated, info = env.step(action)
                    history.append({"type": "state", "content": observation.return_as_json()})
                break
        item["prediction"] = observation.answer
        item["history"] = history
        item["action_cache"] = action_cache

        with open(os.path.join(save_dir, f"{index}.json"), "w") as f:
            json.dump(item, f, indent=4)