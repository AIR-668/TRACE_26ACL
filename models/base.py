# models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class TraceLLM(ABC):
    """统一的 LLM 接口，用于 TRACE 项目"""

    @abstractmethod
    def encode(self, prompts: List[str]) -> Dict[str, Any]:
        """
        输入: 一批 prompt（已经是你构造好的 LogicNLI / ProntoQA 等自然语言问题）
        输出: 
          {
            "hidden_states": List[Tensor],  # len = num_layers+1 (含 embedding 层)
            "tokens": List[List[str]],      # 每个样本对应的 token 序列（可选）
            "model_name": str,
            "meta": {...},
          }
        """
        raise NotImplementedError