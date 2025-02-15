import re
import json
import torch
from rag_gym import State, Action, LLMEngine

class LMReranker:
    
    def __init__(
        self, 
        rerank_llm_name: str = "OpenAI/gpt-4o",
        cache_dir: str | None = None,
        api: bool = False,
        model_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
        **kwargs
    ):
        self.rerank_llm_name = rerank_llm_name
        self.cache_dir = cache_dir
        self.api = api
        self.model_dtype = model_dtype
        self.max_length = max_length
        self.reranker = LLMEngine(
            llm_name = rerank_llm_name,
            cache_dir = cache_dir,
            api = api,
            shared_checkpoint = True
        )

    def __call__(self, state: State, actions: list[Action], next_states: list[State] | None = None, qa_cache: dict[str, str] | None = None):
        question = state.question
        if qa_cache is None:
            curr_history = json.dumps([{"query": query["query"], "documents": query["documents"]} for query in state.history.return_as_json(return_documents=True)], indent=2)
        else:
            curr_history = json.dumps([{"query": query["query"], "answer": qa_cache[query["query"]][-1]} for query in state.history.return_as_json(return_documents=True)], indent=2)
        action_texts = []
        for action in actions:
            if action.query is None:
                action_text = f'Answer: {action.answer}'
            else:
                action_text = f'Query: {action.query}'
            action_texts.append(action_text)
        actions_text = "\n".join([f"[{act_idx}] {action_text}" for act_idx, action_text in enumerate(action_texts)])
        messages = [
            {
                "role": "system",
                "content": "You are a decision-evaluation assistant. Your task is to rank the proposed actions from the most appropriate to the least appropriate as the next step in a sequential decision-making process aimed at solving a given question." 
            },
            {
                "role": "user",
                "content": (
                    f"## Original Question:\n{question}\n\n"
                    f"## Information-Seeking History:\n{curr_history}\n\n"
                    f"## Proposed Next Actions:\n{actions_text}\n\n"
'''### Important Assumption
The agent has no prior knowledge about the subject matter. It must rely solely on the information-seeking history provided to evaluate and answer the original question. Assumptions not explicitly supported by the history must not influence the ranking of proposed actions.

### Evaluation Criteria for Appropriateness

1. **Sufficiency Check**:
- Determine whether the available information is sufficient to directly answer the original question. If not, the proposed action to "Answer" is inappropriate.
- Prioritize queries that gather specific, missing information essential to solving the question.
- If the history already contains all necessary information, then "Answer" is the most appropriate action, and the correct answer should be ranked highest.

2. **Utility Check**:
- Queries must be precise, actionable, and directly relevant to solving the question.
- Prioritize foundational queries that establish critical context or general knowledge necessary for more specific follow-ups.
- Rank overly narrow or prematurely specific queries lower if they presume knowledge not yet available.
- Avoid irrelevant queries that do not contribute to solving the original question.

3. **Redundancy Check**:
- Queries that duplicate information already covered in the history or repeat previous queries should be ranked lower.
- Proposed actions must add new value to the decision-making process by seeking new or clarifying missing information.

### Expected Output Format
- Output the indices of the ranked actions in JSON format: ```json{"ranked_indices": [list of indices]}```.
- Rank actions from most appropriate to least appropriate based on the evaluation criteria above.
- Do not provide additional explanations or reasoning.'''
                )
            }
        ]
        try:
            ranking_text = self.reranker.generate(messages)[0]
            ranked_indices = eval(re.findall(r'```json\s*({.*?})\s*`?', ranking_text, re.DOTALL)[0])["ranked_indices"]
            rewards = [1 / (1 + ranked_indices.index(act_idx)) for act_idx in range(len(actions))]
        except:
            ranked_indices = [act_idx for act_idx in range(len(actions))]
            rewards = [0] * len(actions)
        return rewards