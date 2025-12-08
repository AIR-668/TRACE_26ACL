# models/hf_causal.py
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import TraceLLM

class HFTraceCausalLM(TraceLLM):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_length: int = 2048,
    ):
        self.model_name = model_name
        self.device = device

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # 对很多 causal LLM，需要手动设置 pad_token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,           # 可以是 "auto"
        )
        self.model.eval()

        self.max_length = max_length

    @torch.no_grad()
    def encode(self, prompts: List[str]) -> Dict[str, Any]:
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        outputs = self.model(
            **batch,
            output_hidden_states=True,     # 关键：打开层输出
            use_cache=False,
        )

        hidden_states = outputs.hidden_states  # tuple: (num_layers+1, B, T, D)

        # 转成 list，避免直接挂着大 tensor 到处传
        hs_list = [h.detach().cpu() for h in hidden_states]

        tokens = [
            self.tokenizer.convert_ids_to_tokens(ids.tolist())
            for ids in batch["input_ids"]
        ]

        return {
            "hidden_states": hs_list,
            "tokens": tokens,
            "model_name": self.model_name,
            "meta": {
                "num_layers": len(hs_list) - 1,
                "seq_lens": batch["attention_mask"].sum(dim=-1).tolist(),
            },
        }