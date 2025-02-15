import json
import torch
from rag_gym import State, LLMEngine, Action
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BaseAgent:
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        reward_llm_name: str | None = None,
        train_mode: bool = False
    ):
        self.llm_name = llm_name
        self.cache_dir = cache_dir
        self.api = api
        self.model_dtype = model_dtype
        self.llm = LLMEngine(llm_name=self.llm_name, cache_dir=self.cache_dir, api=self.api, model_dtype=self.model_dtype)
        self.reward_llm_name = reward_llm_name
        self.reward_model = None
        self.reward_tokenizer = None
        self.train_mode = train_mode
        if self.reward_llm_name:
            from peft import PeftModel
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_llm_name, 
                device_map = "auto", 
                cache_dir = cache_dir, 
                torch_dtype = torch.bfloat16,
                num_labels = 1
            )
            self.reward_model = PeftModel.from_pretrained(self.reward_model, reward_llm_name)
            self.reward_model = self.reward_model.merge_and_unload()
            self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_llm_name, cache_dir=cache_dir)
            self.reward_tokenizer.padding_side = 'left'
            # self.reward_tokenizer.padding_side = 'right'
            self.reward_tokenizer.truncation_side = 'left'
            if self.reward_tokenizer.pad_token is None:
                self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
                # self.reward_tokenizer.pad_token = "<|end_of_text|>" # 128001
            if self.reward_model.config.pad_token_id is None:
                # self.reward_model.config.pad_token_id = self.reward_tokenizer.pad_token_id
                self.reward_model.config.pad_token_id = -1

    def generate_action(self, state: State, max_new_tokens: int, temperature: float, num_actions: int) -> list[Action]:
        raise NotImplementedError
    
    def post_process(self, action_str: str) -> Action:
        raise NotImplementedError
    
    def apply_template(self, state: State) -> list[dict[str, str]]:
        raise NotImplementedError
    
    def score(self, state: State, actions: list[Action], max_length=4096, **kwargs):
        assert self.reward_model is not None
        messages = self.apply_template(state, **kwargs)
        all_messages = [messages + [{"role": "assistant", "content": act.action_string}] for act in actions]
        inputs = [self.reward_tokenizer.apply_chat_template(m, tokenize=False) for m in all_messages]
        inputs = self.reward_tokenizer(inputs, add_special_tokens=False, return_tensors="pt", padding="longest", truncation=True, max_length=max_length)
        # inputs = ["\n\n".join([f"{m['role']}: {m['content']}" for m in cm]) for cm in all_messages]
        # inputs = self.reward_tokenizer(inputs, return_tensors="pt", padding="longest", truncation=True, max_length=max_length)
        rewards = self.reward_model(
            input_ids=inputs["input_ids"].to(self.reward_model.device),
            attention_mask=inputs["attention_mask"].to(self.reward_model.device),
            return_dict=True,
        )["logits"]
        return rewards[:,0].tolist()