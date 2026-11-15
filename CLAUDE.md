# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

个人 AI 学习仓库（非生产工程），从 Python 基础 → PyTorch → 大模型应用工程逐步进阶。目录**按 AI 生产流程组织**（`p2_dataset` 数据 → `p4_finetune` 微调 → `p5_inference` 推理 → `p6_rag`/`p7_agent` 应用 → `p8_governance` 治理；`p3_pytorch` 是③微调的前置工具层），当前推进至 RAG（`p6_rag`）。参考资料：`p0_book/AI工程：大模型应用开发实战/`。全部代码注释与文档为中文。

## 目录结构

- `p1_python_basic/` — Python 基础语法练习，按阶段分目录（`P1_基础语法/` …），根下另有 `0_依赖管理.md`（pip 镜像源笔记，用清华源）
- `p2_dataset/` — ② 数据工程（语料清洗、指令对构造）
- `p3_pytorch/` — PyTorch 学习（③微调的前置工具），按主题分目录（`p1_dataloader/` …），每个主题目录含：练习脚本 `.py` + 笔记 `.md` + `case/` 实战案例
- `p4_finetune/` — ③ 模型微调（QLoRA SFT、合并导出、DPO 选学）
- `p5_inference/` — ④ 推理服务：`p1_concepts/`（推理概念，含 `DownloadModel.py`）、`p2_ollama/`（Ollama 笔记）、`p3_quantize/`（量化与 GGUF 转换，含 `基础模型下载.md`）、`p4_vllm/`
- `p6_rag/` — ⑤ 应用·RAG（根下 `note.md` 为 RAG 学习笔记）
- `p7_agent/` — ⑤ 应用·Agent（Prompt/Tool Calling/MCP/Memory/Spring AI）
- `p8_governance/` — ⑥ 治理（Langfuse/评测/护栏/成本）
- `main.py` — PyTorch GPU 环境检测脚本（打印 torch 版本 / CUDA 可用性 / GPU 名）。历史上曾是电商评论脏数据生成脚本，已废弃该用途
- `p0_book/` — 电子书资料
- p3~p8 各层级目录下均有 `README.md` 说明该层定位、子目录作用与学习深度；根 `README.md` 已整合「AI 是怎么工作的」「AI 架构师需要什么能力」「AI 学习路线图」三篇内容

命名约定：目录 `p{n}_{主题}`、脚本 `p{n}_{主题}.py`、阶段目录大写 `P{n}_中文名`。

文件角色约定（p2_dataset 已按此执行，后续层照此办理）：

- 层级 `README.md` 只做**概括提炼与导航**：本层重点、三阶段格式总表、子目录入口、练习路径、进度状态
- 小节 `README.md` 是该小节的**完整教学文件**，固定两部分结构：① 知识学习（由浅到深，含考点）② 课后练习（该节练习题 + 自测题 + 交作业说明），末尾附「待补」清单
- 材料（数据集/造数脚本）放小节目录内，由 README 引用；**不在层根放 note.md / 练习.md 之类散文件**
- **全仓库文档不使用 emoji 表情符号**（含标题装饰、难度星级等，用文字标记如「（考点）」「难度 n/5」）；数据文件里刻意埋的脏数据除外（如 `p3_pytorch/p1_dataloader/case/comments.csv` 的表情刷屏行是清洗练习素材）
- **写作排版规范**：写或改任何 `.md` 文档前，先读四个 skill——`.claude/skills/doc-style/SKILL.md`（中文排版硬规则 + 教学文档骨架）、`.claude/skills/google-dev-style/SKILL.md`（表达规则：结论前置、主动语态、无填充词）、`.claude/skills/tutorial-writing/SKILL.md`（教学设计：目标先行、概念锚点、练习验收标准、检索式自测，写教学文档和练习题时必读）、`.claude/skills/curriculum-design/SKILL.md`（课程设计：逆向设计、学习地图、先修依赖、螺旋加深，规划层结构和学习路线时必读）

## 环境

- Windows 11，用 IDEA/PyCharm 开发，**无项目级 venv**（`.venv`/`venv` 不存在）
- 本仓库 Claude 会话的 shell 中 `python`/`py` 不可用（Windows Store 别名劫持），验证脚本需用户在 IDEA 内运行，或先解决解释器路径
- GPU：GTX 1660 SUPER（8G 显存），torch 2.14.1+cu126，CUDA 可用（以 `main.py` 实测输出为准）
- 大模型下载走 HF 镜像：`HF_ENDPOINT=https://hf-mirror.com`（必须**在 import transformers 之前**设置，见 `p5_inference/p1_concepts/DownloadModel.py`）
- 依赖清单见 `requirements.txt`：torch、torchvision、transformers(==4.46.3)、accelerate、datasets、huggingface_hub、pandas、numpy。**注意该文件是 UTF-16 编码**（git 视为二进制，工具读取显示乱码），编辑前先转 UTF-8

## 大模型推理的已知考点（p5_inference/p1_concepts/DownloadModel.py）

- 8G 显存跑 1.5B 模型的显存预算：`device_map="auto"` + `max_memory={0: "4GB", "cpu": "2GB"}` 显式限制，防止系统被吃满
- `low_cpu_mem_usage=True` 流式加载（边读盘边塞显卡，不占满内存）；`torch_dtype=torch.float16` 半精度
- 推理前 `model.eval()`；对话用 `tokenizer.apply_chat_template` 组装 messages，`generate` 后按输入长度切片再 `batch_decode`

## 数据集

`p3_pytorch/my_datasets/` 被 gitignore，不入库。本地缺失时：

- hymenoptera（蚂蚁/蜜蜂图像）：运行 `p3_pytorch/p1_dataloader/p1_dataset.py` 中的 `snapshot_download` 段，或自行下载
- 电商评论 CSV：旧版 `main.py`（见 git 历史）才有生成逻辑，现行版本已改为 GPU 检测，`case/` 练习如需 CSV 需从历史版本找回

## 学习代码的已知考点（改代码前先看）

- 图片进 DataLoader 前必须统一尺寸（Resize 224×224）并转 RGB，否则 batch 打包报形状错误（见 `p1_dataset.py` 的 compose 流水线）
- 变长文本必须自定义 `collate_fn` 动态 padding，这是 `case/练习.md` 的核心考点（`ReviewDataset.py` 目前 `__getitem__` 未完成，属练习中状态，不是 bug）
- 学习脚本常含大段注释掉的尝试代码，是刻意保留的学习痕迹，清理前先问用户
