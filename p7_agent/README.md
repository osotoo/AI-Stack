# p7_agent — 应用工程·Agent 智能体（路线图 ⑤）

让模型从"会说"到"会做"：调工具、接协议、记上下文、多步规划。与 `p6_rag` 同属应用层，Spring AI 是 Java 技术栈的核心落点。

## 流程位置

- **上游**：`p5_inference` 的模型 API；知识增强可复用 `p6_rag` 的检索管道
- **下游**：`p8_governance` 对调用链做追踪与评测

## 子目录（= 学习顺序）

| 子目录 | 内容 | 作用 |
|---|---|---|
| `p1_prompt/` | Prompt 工程 | system prompt / few-shot / 结构化输出，一切应用层的地基 |
| `p2_tool_calling/` | Tool Calling | 让模型决定何时调哪个函数（查工单、查库等） |
| `p3_mcp/` | MCP 协议 | 工具接入的事实标准（Claude Code 即其最大落地） |
| `p4_memory/` | 会话记忆 | 多轮上下文管理，Redis 存会话历史 |
| `p5_spring_agent/` | Spring AI 编排 | ChatClient / Advisor / ReAct 多步循环，整合成完整智能体 |

## 学习深度要求

- **必须动手**：Tool Calling、Memory 各跑通一个 demo，`p5_spring_agent` 整合出"一个问题同时触发检索 + 工具调用"的效果
- **概念即可**：ReAct 规划模式、多智能体协作

## 当前状态

- 待从零开始
- 两个衔接点：Langfuse 埋点在本层做 `p5_spring_agent` 时就可接入（见 `p8_governance/p1_langfuse`）；`p4_finetune` 微调出的模型部署后，在本层替换现成模型完成贯通闭环
