import re
import json
import torch
from liquid import Template
from liquid.template import BoundTemplate
from rag_gym import State, LLMEngine, Action, BaseAgent

rag_system_prompt = '''You are a helpful assistant. Your task is to think step-by-step and answer a given question using the provided relevant documents.'''

rag_user_template = Template('''### Relevant Documents
{{documents}}

### Question
{{question}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
  - Think step-by-step and then answer the question using the provided relevant documents.

2. **### Structured Output**:
  - Present your predicted answer in the following JSON format:
  ```json
  {
    "predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here."
  }
  ```''')

class RAGAgent(BaseAgent):
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        system_prompt: str | None = None, 
        user_template: BoundTemplate | None = None, 
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        reward_llm_name: str | None = None,
        train_mode: bool = False,
        **kwargs
    ):
        super().__init__(llm_name, cache_dir, api, model_dtype, reward_llm_name, train_mode)
        self.agent_type = "rag"
        self.system_prompt = system_prompt if system_prompt is not None else rag_system_prompt
        self.user_template = user_template if user_template is not None else rag_user_template

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
        if len(state.history) == 0:
            return [Action(query=state.question)]
        action_strings = []
        try:
            action_strings = self.llm.generate(
                messages = self.apply_template(state), 
                max_new_tokens = max_new_tokens, 
                temperature = temperature, 
                num_return_sequences = num_actions, 
                **kwargs
            )
            actions = []
            for action_str in action_strings:
                action = self.post_process(action_str)
                actions.append(action)
        except Exception as E:
            error_class = E.__class__.__name__
            actions = [Action(answer=f"{error_class}: {str(E)}", action_string=f"{error_class}: {str(E)}")]
        return actions
    
    def post_process(self, action_str):
        match = []
        try:
            match = re.findall(r'```json\s*({(?:[^`]|\`(?!``))*})', action_str.split("### Structured Output")[-1], re.DOTALL)
            match = match if len(match) > 0 else re.findall(r'{.*?}', action_str.split("### Structured Output")[-1], re.DOTALL)
            output = eval(re.sub(r' //.*', '', match[-1].replace("null", "None"))) # remove comments
            answer = output["predicted_answer"]
            action = Action(answer=str(answer), action_string=action_str)
        except:
            action = Action(answer=action_str.split("predicted_answer")[-1], action_string=action_str)
        return action

    def apply_template(self, state):
        question = state.question
        documents = '\n'.join(["Document [{:d}] (Title: {:s}) {:s}".format(idx, state.history[0]["documents"][idx]["title"], state.history[0]["documents"][idx]["content"]) for idx in range(len(state.history[0]["documents"]))])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.render(question=question, documents=documents)}
        ]
        return messages