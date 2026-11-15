# p2_instruction — 指令对构造（SFT 数据）

**本节定位**：本层的**核心动手节 + 验收节**。任务：亲手构造 **100 条指令对**（Excel → JSONL），验收标准是**能被 `datasets` 库加载**。产物是 `p4_finetune` QLoRA SFT 的直接训练输入。

## 一、知识学习

### 1.1 SFT 数据长什么样（考点）

- SFT（监督微调）需要 **（指令，响应）** 格式的数据，数据量按**样本数**衡量（预训练按 token 数——这是两阶段的第一处分野）
- 本仓库用 Alpaca 风格三元组，JSONL 每行一条：

```json
{"instruction": "帮我查一下订单状态", "input": "订单号 2026031102358", "output": "该订单已发货，预计…"}
```

- `instruction`：要模型干什么；`input`：补充材料（可为空串）；`output`：期望回答
- **与 chat messages 的关系**：训练时由训练脚本调用 `tokenizer.apply_chat_template` 把三元组转成 `messages` 对话格式——数据侧只管把三元组写对

### 1.2 核心原则：output = 想让模型学会的行为

构造任何一条数据前，先回答：**我希望微调后的模型学会什么行为？**

- 想教 **CoT 分步推理** → output 里就要有分步过程。《Scaling Instruction-Finetuned Language Models》（Chun 等，2024）：微调数据加入分步响应，CoT 任务表现显著提升，某些任务**准确率接近翻倍**
- 想教**工具使用** → output 里要有工具调用过程
- 代价：写分步响应比直接给答案费事得多——这正是数据工作的价值所在
- 反面例子：output 只给最终结论 → 模型学会的是"跳步骤、拍脑袋"

### 1.3 构造流水线（本节主线）

```
Excel 模板 → 逐条填写 → 校验 validate() → 导出 JSONL → datasets 加载 → 覆盖度报告
```

- **为什么用 Excel**：业务同学好上手，收集阶段最常用的协作格式；脚本负责校验和格式转换
- **校验 `validate(df)`**：三个核心字段非空、`output` 长度 ≥ 10、无"待补充/TODO/xxx"占位符、`task_type` 在白名单内
- **`task_type` 先定 4~5 类**（如：事实问答 / 流程操作 / 排查诊断 / 话术生成）——这是后面统计**覆盖度**的依据
- **覆盖度报告**：样本总数、`features` 字段类型、instruction 长度分布、各 task_type 条数分布

### 1.4 合成数据

- 高质量真实数据获取难度大 → 团队普遍转向**合成数据**；直到 AI 能生成逼真且复杂的数据，合成才真正可行
- **合成数据用于训练前同样必须过质量评估**，而且评估 AI 生成的数据本身就很棘手——两道关：程序化质检（复用 validate + 与已有数据去重 + JSON 解析失败丢弃）+ 人工抽检
- 本节用本地 Ollama `qwen2.5:1.5b` 当打工人：1.5B 生成质量一般——**这正好是重点**，亲身体会"合成数据必须过质量评估"的分量
- 合成产物除了补量，还是 `p3_preference`（偏好对）的原料

### 1.5 三要素在本节的落地

| 要素 | 落地动作 |
|---|---|
| 数量 | 100 条起步（验收线） |
| 质量 | `validate()` 程序校验 + 人工抽检；SFT 对噪声容忍度最低，**宁缺毋滥** |
| 覆盖度 | task_type 分布均匀、长度分布合理，别 100 条全是同一类 |

### 1.6 本节工具

| 工具 | 用途 | 备注 |
|---|---|---|
| Python `json` 标准库 | JSONL 读写 | 练习 0 先用它找格式感觉 |
| pandas + `openpyxl` | Excel 模板生成与读取 | **openpyxl 需 `pip install`**，requirements.txt 里没有 |
| HF `datasets` | 加载 JSONL、切分、落盘 | `load_dataset("json", data_files=...)` |
| Ollama HTTP API | 合成数据 | `POST http://localhost:11434/api/generate` |

## 二、本节材料与脚本

由练习逐步产出（数据主题 = 你的工作领域，素材样本待补）：

| 文件 | 产生于 | 说明 |
|---|---|---|
| `instruction_template.xlsx` | 练习 2 | 模板：`instruction/input/output/task_type/备注` |
| `instruction_20.jsonl` | 练习 2 | 第一批 20 条 |
| `sft_dataset.jsonl` | 练习 3 | **验收产物**：100 条 |
| `sft_dataset_disk/` | 练习 3 | `save_to_disk` 切分结果（train/test） |

## 三、课后练习

### 练习 0：JSONL 热身（难度 1/5）

**业务背景**：先不碰任何第三方库，纯手工建立"指令对"的格式感觉。格式感觉没建立，后面全是灾难。

**技术要求**（`p2_instruction/p0_warmup.py`）：
1. 用 Python 列表字典构造 **5 条**指令对（字段：`instruction`、`input`、`output`）。
2. 逐条写入 `warmup.jsonl`（每行一个 JSON 对象）。
3. 再逐行读回来，`json.loads` 解析并打印第 2 条的 `output`。

**验收**：记事本打开是 5 行，每行独立 JSON；程序能正确读回。

### 练习 2：指令对 Excel 模板与校验（难度 3/5）

**业务背景**：团队约定用 Excel 收集指令对（业务同学好上手），你负责设计模板 + 写校验转换脚本。

**技术要求**（`p2_instruction/p2_build.py`）：
1. 先装依赖：`pip install openpyxl`。
2. 生成 `instruction_template.xlsx`：表头 `instruction | input | output | task_type | 备注`，预填 1 行示例。
3. 按**你的工作领域**填入 **≥20 条**真实指令对（这是 100 条的第一批，认真写）。
4. 写校验函数 `validate(df)`（校验规则见 1.3），不合格的行打印行号和原因。
5. 校验通过的行导出 `instruction_20.jsonl`。

### 练习 3：验收项目——100 条指令对 + datasets 加载（难度 4/5）

**业务背景**：本层验收标准。前两个练习的成果在这里汇总。

**技术要求**（`p2_instruction/p3_finalize.py`）：
1. 指令对凑满 **100 条**（Excel 继续填，四个任务类型都要有覆盖）。
2. 全量校验（复用 `validate`），导出 `sft_dataset.jsonl`。
3. `load_dataset("json", data_files="sft_dataset.jsonl", split="train")` 加载。
4. 完成覆盖度报告（见 1.3），`train_test_split(test_size=0.1, seed=42)` 切分并 `save_to_disk`。

**验收（README 原话）**：能被 `datasets` 库加载。跑通即验收通过。

### 练习 4：本地大模型合成指令对（难度 5/5）

**技术要求**（`p2_instruction/p4_synth.py`）：
1. 写 prompt 模板：领域 + 任务类型 → 一次生成 5 条指令对，输出严格 JSON 数组。
2. `requests.post` 调 Ollama 批量生成，目标 50 条。
3. **程序化质检**：复用 `validate` + 与已有 100 条去重 + JSON 解析失败丢弃，统计"生成 50 条最后剩几条"。
4. **人工抽检**：随机抽 10 条标"可用/不可用 + 原因"，结论写进本 README 或笔记。
5. 合成产物留存，给 `p3_preference` 当原料。

### 本节自测（先自己答，再翻回上文）

1. （L5）想让模型学会**分步排查故障**，output 应该长什么样？只给结论的 output 会教出什么模型？
2. （L6）"自动验证数据质量比自动生成数据更难"——用你工作里的例子说明为什么。
3. （L8）合成数据为什么直到最近才真正可行？训练前必须做什么？
4. （L9）JSONL 和一个大 JSON 文件的区别？训练数据为什么惯用 JSONL？
5. （L10）`datasets` 加载本地 JSONL 的写法？加载后怎么看前 3 条 / 字段类型 / 总数？
6. （L11）pandas 读写 xlsx 为什么需要 openpyxl？Excel → JSONL 要经过哪几步？
7. （L12）instruction/input/output 三元组和 `apply_chat_template` 要的 messages 是什么关系？谁负责转换？

**交作业**：`.py` 放本目录 + 发控制台输出（尤其覆盖度报告）+ 说踩坑点。

## 四、待补

- [ ] PDF 第八章：数据合成方法细节（程序化合成 / 模型合成 / 筛选过滤）
- [ ] PDF 第八章：工具使用类数据的样例
- [ ] 你的工作领域素材样本（第一批 5 条范例打磨）
