# p5_inference — 推理服务（路线图 ④）

把模型权重变成**可调用的 API 服务**。本层一半是工具、一半是概念——采样参数、KV Cache、量化这些"懂流程"的关键概念都在这层。

## 流程位置

- **上游**：`p4_finetune` 的模型权重，或直接用现成开源模型
- **下游**：`p6_rag` / `p7_agent` 通过 OpenAI 兼容 API 调用

## 子目录（顺序有讲究）

| 子目录 | 内容 | 作用 |
|---|---|---|
| `p1_concepts/` | 推理概念 | tokenizer / 采样参数（temperature、top_p）/ KV Cache / chat template，用 transformers 直推学原理 |
| `p2_ollama/` | Ollama 部署 | 本地轻量常驻服务，直接 `pull` 现成模型即可跑 |
| `p3_quantize/` | 量化与 GGUF 转换 | 学会转换才能把**任意 HF 模型**（含自己微调的）自部署到 Ollama |
| `p4_vllm/` | vLLM 生产级服务 | 高并发场景 + OpenAI 兼容 API |

推荐路径：直推懂概念 → Ollama 跑现成模型 → 量化自部署 → vLLM 上生产。

## 学习深度要求

- **必须动手**：Ollama 部署、GGUF 量化转换、vLLM 起一次服务
- **概念即可**：KV Cache、PagedAttention、量化位宽（Q4/Q8）与显存换算

## 当前已有内容

- `p1_concepts/DownloadModel.py` — Qwen2.5-1.5B-Instruct 下载与推理（含 8G 显存预算：`device_map="auto"` + `max_memory` 限制、fp16、流式加载）
- `p2_ollama/note.md` — Ollama 完整学习笔记（安装、三种跑法对比、显存预算）
- `p3_quantize/基础模型下载.md` — HF 下载 Base 模型 → llama.cpp 转 GGUF → Ollama 运行的实战指南（含踩坑记录）
