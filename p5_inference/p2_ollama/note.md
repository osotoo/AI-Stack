# Ollama 学习笔记

> 一句话定位：模型是"货物"，Ollama 是本地的"物流公司"——加载、管理、常驻服务一条龙。
> 类比：**Ollama 之于模型 ≈ Docker 之于镜像**。

## 1. Ollama 是什么

- 本地大模型运行工具，底层是 llama.cpp 引擎
- Windows / macOS / Linux 原生支持，NVIDIA 显卡自动走 CUDA
- 启动后是**常驻服务**，默认端口 `11434`，暴露 **OpenAI 兼容 API**

三种跑模型方式对比：

| | transformers 裸跑脚本 | Ollama | vLLM |
|---|---|---|---|
| 生命周期 | 跑完进程退出，模型释放 | 常驻，随叫随到 | 常驻 |
| 定位 | 学原理 | 个人 / 开发用 | 生产级高并发 |
| 上手成本 | 要写代码 | 一条命令 | Linux 环境，较折腾 |

## 2. 使用场景

- **本地学习 / 调试**：一条命令跑起来看效果
- **数据隐私**：敏感数据不出本机
- **给应用当本地 LLM 后端**：OpenAI 兼容接口，应用侧代码几乎零改动
- ❌ 不适合高并发生产环境 → 那是 vLLM 的活

## 3. 安装（Windows 原生，不需要 WSL）

```bash
winget install Ollama.Ollama   # 或官网下载 OllamaSetup.exe
ollama --version               # 验证
```

要点：

- 装完自动注册后台服务，开机自启
- 模型默认存 `C:\Users\<用户>\.ollama\models`，C 盘紧张可用环境变量 `OLLAMA_MODELS` 改到其他盘
- 8G 显存预算参考：1.5B / 3B 随便跑，7B（默认 4bit 量化 ≈4.5G）可以，14B 勉强

## 4. 核心使用

### 4.1 命令行（模型管理）

```bash
ollama pull qwen2.5:1.5b    # 只下载不进入对话
ollama run qwen2.5:1.5b     # 下载（若无）+ 进入对话
ollama list                 # 本地已有模型
ollama ps                   # 当前运行中的模型 + 显存占用
ollama show qwen2.5:1.5b    # 模型详情
ollama rm qwen2.5:1.5b      # 删除模型
```

对话内常用命令：`/bye` 退出、`/clear` 清空上下文、`/set temperature 0.7` 调参、`/?` 查看全部。

### 4.2 API 调用（核心卖点：常驻服务）

OpenAI 兼容端点，第三方 SDK 可无缝接入：

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"你好"}]}'
```

Python 复用 openai SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

### 4.3 与 DownloadModel.py 的对照

同一个 Qwen2.5-1.5B，两种形态：

- **脚本裸跑**：每次运行现加载 → 推理 → 释放。适合搞懂加载、显存、生成的底层原理
- **Ollama**：模型常驻显存 + HTTP 服务常开。适合当应用的推理后端

---

## 安装基础模型

```shell
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir D:\workspace\elsen-space\ai-stack\p0_model\Qwen2.5-1.5B-Base
```