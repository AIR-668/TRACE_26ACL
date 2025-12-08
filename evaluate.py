#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate / extract layerwise embeddings from open-source LLMs
for TRACE_26ACL project.

核心功能：
- 读取 data/{dataset}/{split}.json
- 调用 models/ 里的开源 LLM（如 HFTraceCausalLM）
- 对每个样本提取每一层的 embedding（默认取最后一个 token）
- 保存到 results/embeddings/{dataset}/{model_short}/ex_{id}.npz
- 使用 wandb 记录实验配置 + 简单统计信息
"""

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
import wandb

# ====== 你需要先实现 models/base.py 和 models/hf_causal.py ======
# 这里假设你已经根据我上一条消息创建了 HFTraceCausalLM
from models.hf_causal import HFTraceCausalLM


# -----------------------------
# 1. 数据集读取函数
# -----------------------------

def load_jsonl_or_json(path: str) -> List[Dict[str, Any]]:
    """尽量兼容 JSON / JSONL / 含 data 字段的格式."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # JSONL
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    # 普通 JSON
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # 常见格式：{"data": [...]}
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        # 其他情况直接塞到 list 里
        return [obj]

    raise ValueError(f"Unsupported JSON format in {path}")


def load_dataset(dataset: str, split: str = "dev") -> List[Dict[str, Any]]:
    """
    目前读取：
        data/LogicNLI/dev.json
        data/ProntoQA/dev.json
        data/ProofWriter/dev.json
    如有 train/test，可扩展 split 名字。
    """
    base_dir = os.path.join("data", dataset)
    filename = f"{split}.json"
    path = os.path.join(base_dir, filename)
    print(f"[INFO] Loading dataset from: {path}")
    data = load_jsonl_or_json(path)
    print(f"[INFO] Loaded {len(data)} examples.")
    return data


# -----------------------------
# 2. 把样本转成需要送进 LLM 的文本
# -----------------------------

def build_input_text(dataset: str, ex: Dict[str, Any]) -> str:
    """
    按 dataset 构造 LLM 的输入文本。
    ⚠️ 这里强烈建议你根据自己数据实际字段做微调！
    我先给一个“保底版”，能跑，但可能不是你最终想要的 prompt。
    """

    if dataset == "LogicNLI":
        # TODO: 根据你的 LogicNLI 实际字段修改
        premise = ex.get("premise", ex.get("context", ""))
        hypothesis = ex.get("hypothesis", ex.get("query", ""))
        text = (
            "You are a logical reasoner.\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
        )
        return text

    elif dataset == "ProntoQA":
        # TODO: 根据 ProntoQA 格式修改
        context = ex.get("context", "")
        question = ex.get("question", ex.get("query", ""))
        text = (
            "You are a logical reasoner.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
        )
        return text

    elif dataset == "ProofWriter":
        # TODO: 根据 ProofWriter 格式修改
        facts = ex.get("facts", ex.get("context", ""))
        query = ex.get("query", ex.get("question", ""))
        text = (
            "You are a logical reasoner.\n"
            f"Facts: {facts}\n"
            f"Query: {query}\n"
        )
        return text

    # fallback：如果以上都不匹配，尝试常见字段
    for key in ["input", "question", "query", "text"]:
        if key in ex:
            return str(ex[key])

    # 最底线 fallback：直接 dump 整个样本
    return json.dumps(ex, ensure_ascii=False)


# -----------------------------
# 3. 保存每层 embedding
# -----------------------------

def save_embeddings(
    per_layer_vecs: List[np.ndarray],
    ex_id: str,
    model_name: str,
    dataset: str,
    save_dir: str,
):
    """
    per_layer_vecs: List[np.ndarray]，每个是 (D,) 或 (T, D) 向量
    ex_id: 样本 id（如果数据没有 id 字段，就用索引号）
    """
    model_short = model_name.split("/")[-1].replace(":", "_")

    dir_path = os.path.join(save_dir, dataset, model_short)
    os.makedirs(dir_path, exist_ok=True)

    path = os.path.join(dir_path, f"ex_{ex_id}.npz")
    np.savez_compressed(path, *per_layer_vecs)


# -----------------------------
# 4. 模型构建
# -----------------------------

def build_model(
    model_provider: str,
    model_name: str,
    device: str,
    dtype: str,
    max_length: int,
) -> Any:
    """
    目前只实现 HuggingFace causal LLM 后端。
    如果你之后想加别的 provider，可以在这里扩展。
    """
    if model_provider == "hf_causal":
        return HFTraceCausalLM(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )

    raise ValueError(f"Unknown model_provider: {model_provider}")


# -----------------------------
# 5. 主逻辑：遍历 dataset，取每层 embedding
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    # 数据相关
    parser.add_argument("--dataset", type=str, default="LogicNLI",
                        choices=["LogicNLI", "ProntoQA", "ProofWriter"])
    parser.add_argument("--split", type=str, default="dev")

    # 模型相关
    parser.add_argument("--model_provider", type=str, default="hf_causal",
                        help="目前支持: hf_causal")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="如果 >0，只取前 N 条样本，方便 debug")

    # 保存 & 日志相关
    parser.add_argument("--save_hidden_dir", type=str,
                        default=os.path.join("results", "embeddings"))
    parser.add_argument("--wandb_project", type=str, default="TRACE_26ACL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # 1. 读取数据
    data = load_dataset(args.dataset, args.split)
    if args.max_samples > 0:
        data = data[: args.max_samples]
        print(f"[INFO] Use first {len(data)} samples (max_samples).")

    # 2. 构建模型
    model = build_model(
        model_provider=args.model_provider,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
    )

    # 3. 初始化 wandb
    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.dataset}_{args.split}_{args.model_name.split('/')[-1]}_layers"

    wandb_config = vars(args).copy()
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=wandb_config,
    )

    # 4. 主循环：批量送进 LLM，提取每层 embedding
    num_examples = len(data)
    batch_size = args.batch_size

    print(f"[INFO] Start encoding {num_examples} examples with model {args.model_name}")

    # 用于 wandb 记录简单统计（比如某一层 embedding 的平均 L2 范数）
    layer_norm_sums = None
    layer_norm_counts = 0
    num_layers_plus_one = None  # embedding层 + transformer层数量

    idx = 0
    for start in tqdm(range(0, num_examples, batch_size), desc="Batches"):
        end = min(start + batch_size, num_examples)
        batch = data[start:end]

        prompts = [build_input_text(args.dataset, ex) for ex in batch]

        # 调用模型，拿到所有层 hidden states
        outputs = model.encode(prompts)
        hidden_states = outputs["hidden_states"]  # list[L+1], 每个 (B, T, D)
        model_name = outputs.get("model_name", args.model_name)

        if num_layers_plus_one is None:
            num_layers_plus_one = len(hidden_states)
            print(f"[INFO] Model returned {num_layers_plus_one} hidden states (incl. embedding).")

        # 遍历 batch 内每个样本
        for b, ex in enumerate(batch):
            ex_id = str(ex.get("id", idx))  # 如果样本没有 id 字段，就用 running index

            per_layer_vecs = []
            layer_norms = []

            for layer_idx, h in enumerate(hidden_states):
                # h: (B, T, D) torch.Tensor
                # 取这个样本的序列 (T, D)
                seq_hidden = h[b]  # (T, D)

                # 假设你关心最后一个 token 的表示（类似 CLS），也可以改成平均等
                # 这里可以用 attention_mask 来确定真实长度，这里简化：直接用最后一个位置
                last_vec = seq_hidden[-1]  # (D,)
                vec_np = last_vec.detach().cpu().numpy()
                per_layer_vecs.append(vec_np)

                # 统计 L2 范数，方便 wandb 画图观察各层尺度
                layer_norms.append(float(last_vec.norm(p=2).item()))

            # 保存到文件
            save_embeddings(
                per_layer_vecs=per_layer_vecs,
                ex_id=ex_id,
                model_name=model_name,
                dataset=args.dataset,
                save_dir=args.save_hidden_dir,
            )

            # 更新整体统计
            if layer_norm_sums is None:
                layer_norm_sums = np.zeros(len(layer_norms), dtype=np.float64)
            layer_norm_sums += np.array(layer_norms, dtype=np.float64)
            layer_norm_counts += 1
            idx += 1

        # 可以按 batch 往 wandb 里 log 一次平均 norm
        avg_layer_norms = layer_norm_sums / max(layer_norm_counts, 1)
        log_dict = {
            f"layer_{i}_avg_l2": float(v)
            for i, v in enumerate(avg_layer_norms)
        }
        log_dict["num_processed"] = layer_norm_counts
        wandb.log(log_dict)

    print(f"[INFO] Finished. Total processed examples: {layer_norm_counts}")
    wandb.finish()


if __name__ == "__main__":
    main()