import re
import json
import torch
from liquid import Template
from liquid.template import BoundTemplate
from rag_gym import State, LLMEngine, Action, BaseAgent

react_system_prompt = '''You are a helpful assistant. Your task is to think step-by-step and take an action to help solve a given question.'''

react_user_template = Template('''### Information-seeking History
{{history}}

### Original Question
{{question}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
  - Think step-by-step and reason about the current situation.
  - Take an action for the next step which can be two types:
    - Search[query], which searches the exact query in an external knowledge base. Avoid duplicating queries already asked in the history.
    - Finish[answer], which returns the answer and finishes the task.

2. **### Structured Output**:
  - If the next action is `Search`, present your generated query in the following JSON format:
    ```json
    {
        "generated_query": "Provide an entity, question, or statement to be searched in an external knowledge base.",
    }
    ```
  - If the next action is `Finish`, present your predicted answer in the following JSON format:
    ```json
    {
        "predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here."
    }
    ```''')

react_truncated_user_template = Template('''### Information-seeking History
{{history}}

### Original Question
{{question}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
  - Think step-by-step and reason about the current situation.
  - Take an action for the next step which can be one type:
    - Finish[answer], which returns the answer and finishes the task.

2. **### Structured Output**:
  - Present your predicted answer in the following JSON format:
    ```json
    {
        "predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here."
    }
    ```''')

class ReActAgent(BaseAgent):
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        system_prompt: str | None = None, 
        user_template: BoundTemplate | None = None, 
        truncated_user_template: BoundTemplate | None = None, 
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        reward_llm_name: str | None = None,
        train_mode: bool = False,
        **kwargs
    ):
        super().__init__(llm_name, cache_dir, api, model_dtype, reward_llm_name, train_mode)
        self.agent_type = "react"
        self.system_prompt = system_prompt if system_prompt is not None else react_system_prompt
        self.user_template = user_template if user_template is not None else react_user_template
        self.truncated_user_template = truncated_user_template if truncated_user_template is not None else react_truncated_user_template

    def generate_action(
        self, 
        state: State, 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_actions: int = 1,
        **kwargs
    ):
        '''
            Input: a State object, parameters for generating an action
            Output: a list of Gap objects
        '''
        history = "\n\n".join([f"Query: {item['query']}\n" + "\n".join([f"Document [{idx}] (Title: {doc['title']}) {doc['content']}" for idx, doc in enumerate(item["documents"])]) for item in state.history])
        action_strings = []
        try:
            question = state.question
            action_strings = self.llm.generate(
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_template.render(question=question, history=history) if not state.truncated else self.truncated_user_template.render(question=question, history=history)}
                ], 
                max_new_tokens = max_new_tokens, 
                temperature = temperature, 
                num_return_sequences = num_actions, 
                **kwargs
            )
            actions = []
            for action_str in action_strings:
                assert type(action_str) is str
                action = self.post_process(action_str)
                actions.append(action)
        except Exception as E:
            error_class = E.__class__.__name__
            actions = [Action(action_string=f"{error_class}: {str(E)}")]
        return actions
    
    def post_process(self, action_str):
        match = []
        try:
            match = re.findall(r'```json\s*({(?:[^`]|\`(?!``))*})', action_str.split("### Structured Output")[-1], re.DOTALL)
            match = match if len(match) > 0 else re.findall(r'{.*?}', action_str.split("### Structured Output")[-1], re.DOTALL)
            output = eval(re.sub(r' //.*', '', match[-1].replace("null", "None"))) # remove comments
            query = output.get("query", output.get("generated_query", output.get("generated_queries", None)))
            answer = output.get("predicted_answer", None)
            if type(query) == list:
                query = query[0]
            query = str(query) if query is not None else query
            if query:
                action = Action(query=str(query), action_string=action_str)
            else:
                action = Action(answer=str(answer), action_string=action_str)
        except:
            action = Action(query=action_str.split("### Structured Output")[-1].strip(), action_string=action_str)
        return action

    def apply_template(self, state):
        question = state.question
        history = "\n\n".join([f"Query: {item['query']}\n" + "\n".join([f"Document [{idx}] (Title: {doc['title']}) {doc['content']}" for idx, doc in enumerate(item["documents"])]) for item in state.history])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.render(question=question, history=history)}
        ]
        return messages