# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

个人 AI 学习仓库（非生产工程），从 Python 基础 → PyTorch → 大模型应用逐步进阶。参考资料：`p0_book/AI工程：大模型应用开发实战/AI工程：大模型应用开发实战.pdf`。全部代码注释与文档为中文。

## 目录结构

- `p1_python_basic/` — Python 基础语法练习，按阶段分目录（`P1_基础语法/` …）
- `p2_pytorch/` — PyTorch 学习，按主题分目录（`p1_dataset/` …），每个主题目录含：练习脚本 `.py` + 笔记 `.md` + `case/` 实战案例
- `main.py` — 脏数据生成脚本，产出电商评论 CSV 供 Dataset 清洗练习（注意：其输出目录是 `./p1_dataloader/`，与 `p2_pytorch/p1_dataset/case/` 中的练习并非同一份文件）
- `p0_book/` — 电子书资料

命名约定：目录 `p{n}_{主题}`、脚本 `p{n}_{主题}.py`、阶段目录大写 `P{n}_中文名`。

## 环境

- Windows 11，用 IDEA/PyCharm 开发，**无项目级 venv**（`.venv`/`venv` 不存在）
- 本仓库 Claude 会话的 shell 中 `python`/`py` 不可用（Windows Store 别名劫持），验证脚本需用户在 IDEA 内运行，或先解决解释器路径
- 依赖清单见 `requirements.txt`：pandas、numpy、torch、torchvision、transformers、datasets、huggingface_hub、Pillow（未锁版本，本机安装后如需可 pip freeze 补齐）

## 数据集

`p2_pytorch/*/my_datasets/` 被 gitignore，不入库。本地缺失时：

- hymenoptera（蚂蚁/蜜蜂图像）：运行 `p2_pytorch/p1_dataset/p1_dataset.py` 中的 `snapshot_download` 段，或自行下载
- 电商评论 CSV：由根目录 `main.py` 生成

## 学习代码的已知考点（改代码前先看）

- 图片进 DataLoader 前必须统一尺寸（Resize 224×224）并转 RGB，否则 batch 打包报形状错误（见 `p1_dataset.py` 的 compose 流水线）
- 变长文本必须自定义 `collate_fn` 动态 padding，这是 `case/练习.md` 的核心考点（`ReviewDataset.py` 目前 `__getitem__` 未完成，属练习中状态，不是 bug）
- 学习脚本常含大段注释掉的尝试代码，是刻意保留的学习痕迹，清理前先问用户
