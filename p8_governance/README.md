# p8_governance — 工程纪律与治理（路线图 ⑥）

让 AI 应用**可评估、可观测、可控、可算账**。没有这层，应用只是 demo；对企业级交付而言，护栏与合规是硬性要求。

## 流程位置

- **上游**：`p6_rag` / `p7_agent` 上报的指标、trace 与 badcase
- **回流**：评测发现的 badcase 回流 `p2_dataset`（补数据）和 `p4_finetune`（再训练）——AI 工程是闭环，不是单向管道

## 子目录

| 子目录 | 内容 | 作用 |
|---|---|---|
| `p1_langfuse/` | 链路追踪 | 每次 LLM 调用的完整链路可视化（做 `p7_agent` 时就可先接入，不必等） |
| `p2_eval/` | 评测 | 自建评测集回归 + RAGAS 四指标（Faithfulness / Answer Relevancy / Context Precision / Context Recall） |
| `p3_guardrails/` | 安全护栏 | Prompt 注入防护、输出内容安全（企业级刚需） |
| `p4_cost/` | 成本治理 | Token 计费核算、网关限流 |

注意：RAGAS 是 Python 生态，Java 技术栈的落地方式是 Langfuse Dataset 评测或自建测试集，RAGAS 旁路跑。

## 学习深度要求

- **必须动手**：Langfuse 接入 Spring AI 并看到一条完整 trace；自建 20 条评测集，做到每次改 Prompt/参数都能对比出分数变化
- **概念即可**：RAGAS 四指标含义、限流策略选型

## 当前状态

- 全新领域，从零起步，对应贯通项目最后一步
