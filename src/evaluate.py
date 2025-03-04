import re
import os
import json
import tqdm
import argparse
import sys
sys.path.append("./src")
from hotpot_evaluate_v1 import normalize_answer, f1_score, exact_match_score

def normalize_answer_multichoice(s):
    ans = re.findall("(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("^\s*[\"\']?(A|B|C|D)[$/,\.\"\':]", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("^\s*(A|B|C|D) or", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("^\s*(A|B|C|D) and", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("[Oo]ption (A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0]
    ans = re.findall(":\s*(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall(r"\$?\\boxed\{(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("\*\*[Aa]nswer:?\*\*\s*(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("[Aa]nswer is:?\s*\{?[\"\']?(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("Therefore.*(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("-?-?>\s*\{?[\"\']?(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall(r"is:?[\s\n]*\*?\*?(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    return s.strip()

def locate_answer(s):
    if s is None:
        return "NA"
    s = re.sub('\s+', ' ', s)
    groups = re.search(r"answer_choice[\"\']:\s*\{?[\"\']?(.+?)[\"\']?\s*\}", s)
    if groups:
        return groups.group(1)
    groups = re.search(r"answer[\"\']:\s*\{?[\"\']?(.+?)[\"\']?\s*\}", s)
    if groups:
        return groups.group(1)
    groups = re.search(r"answer is:?\s*\{?[\"\']?(.+?)[\"\']?\s*\}", s)
    if groups:
        return groups.group(1)
    groups = re.search(r"[Aa]nswer\*?\*?:\s*(A|B|C|D)", s)
    if groups:
        return groups.group(1)
    groups = re.search(r"is:?\s*\*?\*?(A|B|C|D)", s)
    if groups:
        return groups.group(1)
    return s.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, default="direct")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--reward_llm_name", type=str, default=None)
    parser.add_argument("--retriever_name", type=str, default=None)
    parser.add_argument("--corpus_name", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--n_actions", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--data", type=str, default="medqa")
    parser.add_argument("--save_dir", type=str, default="./predictions")
    args = parser.parse_args()

    agent_type = args.agent_type
    llm_name = args.llm_name
    reward_llm_name = args.reward_llm_name
    retriever_name = args.retriever_name
    corpus_name = args.corpus_name
    max_iterations = args.max_iterations
    n_actions = args.n_actions if reward_llm_name else 1
    temperature = args.temperature if reward_llm_name else 0.0
    k = args.k
    rrf_k = args.rrf_k
    data = args.data
    save_dir = args.save_dir

    save_dir = os.path.join(save_dir, data, agent_type, '_'.join([name.replace('/', '_') for name in [llm_name, reward_llm_name] if name is not None]), f"{max_iterations}" if n_actions == 1 else f"{max_iterations}_{n_actions}_{temperature}")

    if agent_type in ["rag", "react", "search_o1", "research"]:
        retriever_name = retriever_name if retriever_name else "RRF-2" if data == "medqa" else "RRF-BGE"
        corpus_name = corpus_name if corpus_name else "MedText" if data == "medqa" else "Wikipedia_HotpotQA" if data == "hotpotqa" else "Wikipedia"
        if "rrf" in retriever_name.lower():
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}_{rrf_k}"
        else:
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}"
        save_dir = os.path.join(save_dir, retrieval_suffix)
    
    fnames = [fpath for fpath in sorted(os.listdir(save_dir)) if fpath.endswith(".json") and "history" not in fpath and "qa_cache" not in fpath and "qd_cache" not in fpath]

    count = 0
    error_count = 0
    valid_fnames = []

    results = []
    for fpath in tqdm.tqdm(fnames):
        with open(os.path.join(save_dir, fpath)) as f:
            try:
                item = json.load(f)
            except:
                print(f"Error in {fpath}")
                continue
            if "prediction" not in item and "predictions" not in item:
                continue
            results.append(item)
            count += 1
            if "Error:" in json.dumps(item):
                error_count += 1
            valid_fnames.append(fpath)

    normalize = normalize_answer if data in ["hotpotqa", "2wikimultihopqa", "bamboogle"] else normalize_answer_multichoice

    if data in ["hotpotqa", "2wikimultihopqa", "bamboogle"]:
        metrics = {"em": 0, "f1": 0, "prec": 0, "recall": 0}
    else:
        metrics = {"acc": 0}

    num_actions = []

    na_count = 0

    for item in tqdm.tqdm(results):
        answer = item["answer"]
        
        if item["prediction"] is None:
            na_count += 1
            prev_answers = [action["content"]["answer"] for action in item["history"] if action["content"]["answer"] is not None]
            item["prediction"] = prev_answers[-1] if len(prev_answers) > 0 else None

        if type(item["prediction"]) == dict:
            item["prediction"] = sorted(item["prediction"].values())[0]
        pred = locate_answer(str(item["prediction"]))

        if "history" in item:
            num_actions.append(sum([True for h in item["history"] if h["type"] == "action"]))

        if data == "medqa":
            acc = normalize(pred) == normalize(answer)
            metrics["acc"] += float(acc)
        else:
            em = exact_match_score(pred, answer)
            f1, prec, recall = f1_score(pred, answer)
            metrics["em"] += float(em)
            metrics["f1"] += f1
            metrics["prec"] += prec
            metrics["recall"] += recall

    for key in metrics.keys():
        metrics[key] = f"{metrics[key]/count:.4f}"

    if len(num_actions) > 0:
        print(f"Average number of actions: {sum(num_actions) / len(num_actions)}")
    print(f"Number of samples: {count} | Number of error samples: {error_count} | Number of N/A samples: {na_count} | Metrics: {metrics}")