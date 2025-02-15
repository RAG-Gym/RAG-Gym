import os
import json
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cached_llms = dict()
cached_retrievers = dict()
cached_corpora = dict()

class LLMEngine:
    def __init__(
        self,
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        cache_dir: str | None = None, 
        api: bool = False, 
        lora: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        shared_checkpoint: bool = True,
    ):
        self.llm_name = llm_name
        self.cache_dir = cache_dir or os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        self.api = api
        self.model_dtype = model_dtype
        if api == True and "gpt" in self.llm_name.lower():
            import openai
            from rag_gym import config_openai
            openai.api_type = config_openai.get("api_type") or openai.api_type or os.getenv("OPENAI_API_TYPE")
            openai.api_version = config_openai.get("api_version") or openai.api_version or os.getenv("OPENAI_API_VERSION")
            openai.api_key = config_openai.get("api_key") or openai.api_key or os.getenv('OPENAI_API_KEY')
            self.model_name = self.llm_name.split("/")[-1]
            if openai.api_type == "azure":
                openai.azure_endpoint = openai.azure_endpoint or os.getenv("OPENAI_ENDPOINT") or config_openai.get("api_base")
                self.client = openai.AzureOpenAI(
                    api_version=openai.api_version,
                    azure_endpoint=openai.azure_endpoint,
                    api_key=openai.api_key,
                )
            else:
                self.client = openai.OpenAI(
                    api_key=openai.api_key,
                )
            self.model = None
        elif api == True:
            import openai
            from rag_gym import config_azure
            self.model_name = "azureai"
            self.client = openai.OpenAI(
                base_url=config_azure["base_url"], 
                api_key=config_azure["api_key"],
                default_headers={"extra-parameters": "pass-through"}
            )
            self.model = None
        else:
            global cached_llms
            new_llm = True
            if self.llm_name in cached_llms and shared_checkpoint:
                self.model = cached_llms[self.llm_name]
                new_llm = False
            if new_llm:
                model = AutoModelForCausalLM.from_pretrained(self.llm_name, device_map="auto", cache_dir=self.cache_dir, torch_dtype=self.model_dtype)
                tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
                if os.path.exists(self.llm_name) and os.path.exists(os.path.join(self.llm_name, "adapter_config.json")):
                    lora = True
                if cache_dir and os.path.exists(os.path.join(cache_dir, "--".join(["models"] + self.llm_name.split('/')))) and os.path.exists(os.path.join(cache_dir, "--".join(["models"] + self.llm_name.split('/')), "adapter_config.json")):
                    lora = True
                if lora:
                    from peft import PeftModel
                    # base_model_name = json.load(open(os.path.join(self.llm_name, "adapter_config.json")))["base_model_name_or_path"]
                    model = PeftModel.from_pretrained(model, self.llm_name)
                    model = model.merge_and_unload()
                else:
                    # self.model = transformers.pipeline(
                    #     "text-generation",
                    #     model=self.llm_name,
                    #     torch_dtype=self.model_dtype,
                    #     device_map="auto",
                    #     model_kwargs={"cache_dir":self.cache_dir},
                    # )
                    pass
                self.model = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
                self.model.model.config._name_or_path = self.llm_name
                self.model.model.generation_config.temperature=None
                self.model.model.generation_config.top_p=None
                if shared_checkpoint:
                    cached_llms[self.llm_name] = self.model
            self.client = None

    def generate(
        self, 
        messages: list[dict[str, str]], 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_return_sequences: int = 1,
        **kwargs
    ):
        if self.model is not None:
            if temperature > 0.0:
                response = self.model(messages, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, temperature=temperature, pad_token_id=self.model.tokenizer.eos_token_id, **kwargs)
                outputs = [response[i]["generated_text"][-1]["content"] for i in range(num_return_sequences)]
            else:
                response = self.model(messages, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.model.tokenizer.eos_token_id, **kwargs)
                outputs = [response[0]["generated_text"][-1]["content"]]
        else:
            response = self.client.chat.completions.create(
                model= self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=num_return_sequences if temperature > 0.0 else 1,
                **kwargs
            )
            outputs = [response.choices[i].message.content for i in range(num_return_sequences if temperature > 0.0 else 1)]
        return outputs