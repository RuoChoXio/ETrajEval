# model.py

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from accelerate import infer_auto_device_map, init_empty_weights
from openai import OpenAI, APIError
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# BaseModel remains unchanged
class BaseModel:
    def __init__(self, name: str, generation_params: Dict[str, Any]):
        self.name = name
        self.generation_params = generation_params
    def generate(self, history: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError
    def batch_generate(self, histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        raise NotImplementedError


class LocalModel(BaseModel):
    """
    A wrapper for local Hugging Face transformer models with precise GPU allocation.
    """
    def __init__(self, path_or_name: str, gpu_indices: List[int], name: str, generation_params: Dict[str, Any]):
        super().__init__(name, generation_params)
        print(f"[*] Initializing local model '{name}' on GPU indices: {gpu_indices}")

        self.tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
        
        loading_args = {"trust_remote_code": True}

        if len(gpu_indices) > 1:
            print(f"    - Distributing model across {len(gpu_indices)} GPUs...")
            config = AutoConfig.from_pretrained(path_or_name, trust_remote_code=True)
            with init_empty_weights():
                model_empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            
            max_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in gpu_indices}
            
            no_split_module_classes = getattr(model_empty, "_no_split_modules", [])
            device_map = infer_auto_device_map(
                model_empty,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=torch.bfloat16
            )
            loading_args["device_map"] = device_map

        elif len(gpu_indices) == 1:
            device_str = f"cuda:{gpu_indices[0]}"
            loading_args["device_map"] = {"": device_str}
        else:
            raise ValueError("gpu_indices must contain at least one index.")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            path_or_name,
            torch_dtype=torch.bfloat16,
            **loading_args
        ).eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        print(f"[*] Local model '{self.name}' loaded successfully.")

    def generate(self, history: List[Dict[str, str]], **kwargs) -> str:
        prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        gen_params = {**self.generation_params, **kwargs}
        if "max_tokens" in gen_params:
            gen_params["max_new_tokens"] = gen_params.pop("max_tokens")

        outputs = self.model.generate(**inputs, **gen_params)
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()

    def batch_generate(self, histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        prompts = [self.tokenizer.apply_chat_template(h, tokenize=False, add_generation_prompt=True) for h in histories]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        
        gen_params = {**self.generation_params, **kwargs}
        if "max_tokens" in gen_params:
            gen_params["max_new_tokens"] = gen_params.pop("max_tokens")

        outputs = self.model.generate(**inputs, **gen_params)
        input_lengths = inputs.input_ids.shape[1]
        response_ids = outputs[:, input_lengths:]
        decoded_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        return [resp.strip() for resp in decoded_responses]


class APIModel(BaseModel):
    """
    A wrapper for API-based models like OpenAI's, with retry logic.
    """
    def __init__(self, model_name: str, api_key: str, base_url: str, name: str, generation_params: Dict[str, Any]):
        super().__init__(name, generation_params)
        print(f"[*] Initializing API model: {model_name}")
        resolved_api_key = os.getenv("OPENAI_API_KEY", api_key)
        if not resolved_api_key or resolved_api_key == "YOUR_API_KEY_HERE":
            raise ValueError("API key not found. Please set OPENAI_API_KEY environment variable or define it in config.yaml")
        self.client = OpenAI(api_key=resolved_api_key, base_url=base_url)
        self.model_name = model_name
        print(f"[*] API model '{self.name}' initialized successfully.")

    def generate(self, history: List[Dict[str, str]], **kwargs) -> str:
        gen_params = {**self.generation_params, **kwargs}
        retries = 3 # Number of retries
        
        for attempt in range(retries):
            try:
                # print(history)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    timeout=900,
                    **gen_params
                )
                # print(response)
                # Handle cases where the response content might be None or choices are empty
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                else:
                    print(f"\n[!] Warning for model {self.name}: API returned an empty response. Returning empty string.")
                    return "" # Return empty string for NoneType content
                    
            except APIError as e:
                print(f"\n[!] API Error for model {self.name} on attempt {attempt + 1}/{retries}: {e}")
                time.sleep(10) # Wait before retrying
            except Exception as e:
                print(f"\n[!] An unexpected error occurred for model {self.name} on attempt {attempt + 1}/{retries}: {e}")
                time.sleep(10) # Wait before retrying

        # If all retries fail, return an error message
        print(f"\n[!] All {retries} retry attempts failed for model {self.name}.")
        return ""

    def batch_generate(self, histories: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        
        def run_generate(history_item):
            return self.generate(history_item, **kwargs)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(run_generate, histories))
        return results

# RewardModel class remains unchanged
class RewardModel:
    """
    A wrapper for loading and using a sequence classification model for reward/sentiment scoring.
    """
    def __init__(self, path: str, device: str):
        """
        Loads the reward model onto a specified device.

        Args:
            path (str): The path to the Hugging Face model.
            device (str): The device to load the model on (e.g., "cuda:0").
        """
        print(f"[*] Initializing reward model from '{path}' on device '{device}'...")
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError("CUDA device specified, but CUDA is not available.")
        
        self.device = torch.device(device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, 
                torch_dtype=torch.bfloat16, 
                # device_map={"": self.device},
                device_map="auto",
                num_labels=1
            ).eval()
            print(f"[+] Reward model loaded successfully to {self.model.device}.")
        except Exception as e:
            print(f"[!] FATAL ERROR: Failed to load reward model from '{path}'. Error: {e}")
            raise

    def get_batch_scores(self, batch_chats: List[List[Dict[str, str]]]) -> List[float]:
        """
        Gets raw logit scores for a batch of conversation turns.
        """
        prompts = [
            self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            for chat in batch_chats
        ]
        
        tokenized_batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**tokenized_batch).logits
            return logits.squeeze().cpu().tolist()