# p2_instruction — 指令对构造（SFT 数据）

100 条亲手构造的指令对，是本层的验收作品，也是 `p4_finetune` QLoRA SFT 的直接训练输入。本节解决四个问题：指令对长什么样，output 怎么写才算好，数据不够怎么用本地大模型合成，以及合成完怎么验货。

## 一、知识学习

### SFT 数据长什么样（考点）

SFT（监督微调）需要"指令、响应"成对的数据，数据量按样本数衡量——预训练按 token 数算账，这是两个阶段的第一处分野。本仓库用 Alpaca 风格三元组，JSONL 每行一条：

```json
{"instruction": "帮我查一下订单状态", "input": "订单号 2026031102358", "output": "该订单已发货，预计……"}
```

三个字段的分工：`instruction` 说要模型干什么，`input` 补充材料（可以为空字符串），`output` 是期望回答。

三元组和训练时的 `messages` 对话格式是什么关系？训练脚本调用 `tokenizer.apply_chat_template` 负责转换，数据侧只管把三元组写对，不用操心对话模板。

### output 等于想让模型学会的行为（核心）

构造任何一条数据之前，先回答：我希望微调后的模型学会什么行为？output 就写成那个行为的样子。

最典型的例子是分步推理。想让模型学会一步步排查问题，output 里就要有分步过程——《Scaling Instruction-Finetuned Language Models》（Chun 等，2024）的结论是：微调数据里加入分步响应，CoT 任务表现显著提升，某些任务准确率接近翻倍。反过来，如果 output 只给最终结论，模型学会的就是"跳步骤、拍脑袋"。

代价要说清楚：写分步响应比直接给答案费事得多。这份人工，就是"有多少人工就有多少智能"里的人工。

### 构造流水线：本节主线

```text
Excel 模板 → 逐条填写 → 校验 validate() → 导出 JSONL → datasets 加载 → 覆盖度报告
```

为什么用 Excel？收集阶段最常用的协作格式是它，业务同学上手零成本；校验和格式转换交给脚本。流水线每一环的要求：

- 校验 `validate(df)`：三个核心字段非空、`output` 长度不小于 10、没有"待补充 / TODO / xxx"占位符、`task_type` 在白名单内
- `task_type` 先定 4 到 5 类，比如事实问答、流程操作、排查诊断、话术生成——这是后面统计覆盖度的依据
- 覆盖度报告四样：样本总数、`features` 字段类型、instruction 长度分布、各 task_type 条数分布

### 合成数据：让本地大模型打工

高质量真实数据难拿，行业普遍转向合成数据。合成的可行性是最近才成立的——直到 AI 能生成逼真且复杂的数据，它才真正可用。但有一条铁律：**合成数据用于训练之前，必须过质量评估**，而评估 AI 生成的数据，和评估其他 AI 输出一样棘手。

实操上过两道关：程序化质检（复用 validate、和已有数据去重、JSON 解析失败的直接丢弃），加上人工抽检。本节用本地 Ollama 的 `qwen2.5:1.5b` 当打工人——它的生成质量一般，而这正好是重点：亲身体会"合成数据必须验货"这句话的分量。合成产物除了补量，还是 `p3_preference` 偏好对的原料。

### 三要素在本节的落地

| 要素 | 落地动作 |
|---|---|
| 数量 | 100 条起步（验收线） |
| 质量 | `validate()` 程序校验加人工抽检；SFT 对噪声容忍度最低，宁缺毋滥 |
| 覆盖度 | task_type 分布均匀、长度分布合理，别 100 条全是同一类 |

### 本节工具

| 工具 | 用途 | 备注 |
|---|---|---|
| Python `json` 标准库 | JSONL 读写 | 练习 0 先用它找格式感觉 |
| pandas + openpyxl | Excel 模板生成与读取 | openpyxl 要单独 `pip install`，requirements.txt 里没有 |
| HF datasets | 加载 JSONL、切分、落盘 | `load_dataset("json", data_files=...)` |
| Ollama HTTP API | 合成数据 | `POST http://localhost:11434/api/generate` |

## 二、本节材料与脚本

以下文件由练习逐步产出，数据主题用你的工作领域（素材样本待补）：

| 文件 | 产生于 | 说明 |
|---|---|---|
| `instruction_template.xlsx` | 练习 2 | 模板，表头 `instruction / input / output / task_type / 备注` |
| `instruction_20.jsonl` | 练习 2 | 第一批 20 条 |
| `sft_dataset.jsonl` | 练习 3 | 验收产物：100 条 |
| `sft_dataset_disk/` | 练习 3 | `save_to_disk` 切分结果（train、test） |
| `questions.ipynb` | 加练 | 10 道手感题：JSONL 读写、validate、覆盖度统计，只用标准库加 pandas，Colab 打开即做 |

## 三、课后练习

### 练习 0：JSONL 热身（难度 1/5）

先不碰任何第三方库，纯手工建立"指令对"的格式感觉。格式感觉没建立，后面全是灾难。

在 `p2_instruction/p0_warmup.py` 里：

1. 用 Python 列表字典构造 5 条指令对，字段为 `instruction`、`input`、`output`
2. 逐条写入 `warmup.jsonl`，每行一个 JSON 对象
3. 再逐行读回来，`json.loads` 解析，打印第 2 条的 `output`

验收标准：记事本打开是 5 行，每行是独立 JSON；程序能正确读回。

### 练习 2：指令对 Excel 模板与校验（难度 3/5）

团队约定用 Excel 收集指令对，业务同学好上手；你负责设计模板，写校验转换脚本。原料有两路：你的工作领域，加 p1 洗出来的干净语料。

在 `p2_instruction/p2_build.py` 里：

1. 先装依赖：`pip install openpyxl`
2. 生成 `instruction_template.xlsx`，预填 1 行示例
3. 按你的工作领域填入至少 15 条真实指令对——这是 100 条的第一批，认真写
4. 从 `p1_corpus/clean_corpus.csv` 挑 5 条干净语料，改写成指令对（针对这条语料提任务当 instruction，原文放 input，期望回答当 output）——清洗成果经这条依赖链流进构造
5. 实现校验函数 `validate(df)`，规则见"构造流水线"一节；不合格的行打印行号和原因
6. 校验通过的行导出 `instruction_20.jsonl`

### 练习 3：验收项目——100 条指令对（难度 4/5）

本层的验收标准，前两个练习的成果在这里汇总。100 条手写是本层最重的活，刻意为之：亲手写过才知道痛，痛过才懂练习 4 合成数据的价值。可以分几天填，别一口气。

在 `p2_instruction/p3_finalize.py` 里：

1. 指令对凑满 100 条，四个任务类型都要有覆盖
2. 全量校验（复用 `validate`），导出 `sft_dataset.jsonl`
3. 用 `load_dataset("json", data_files="sft_dataset.jsonl", split="train")` 加载
4. 完成覆盖度报告，然后 `train_test_split(test_size=0.1, seed=42)` 切分，`save_to_disk` 落盘——留 test 的原因：p4 微调后在它上面对比效果，才能证明微调真有用

验收就是一句话：能被 `datasets` 库加载，跑通即通过。

### 练习 4：本地大模型合成指令对（难度 5/5）

手写 100 条之外再翻倍，成本就顶不住了——行业的做法是让模型打工生成草稿，人来验货。这是"合成数据必须过质量评估"的实战现场。

在 `p2_instruction/p4_synth.py` 里：

1. 写 prompt 模板：给领域和任务类型，让模型一次生成 5 条指令对，输出严格 JSON 数组
2. 用 `requests.post` 调 Ollama 批量生成，目标 50 条
3. 程序化质检：复用 `validate`，和已有 100 条去重，JSON 解析失败丢弃；统计"生成 50 条，最后剩几条"
4. 人工抽检：随机抽 10 条，逐条标"可用 / 不可用 + 原因"，结论写进本 README
5. 合成产物留存，给 `p3_preference` 当原料

### 本节自测

1. 想让模型学会分步排查故障，output 应该长什么样？只给结论的 output 会教出什么模型？
2. "自动验证数据质量比自动生成数据更难"——用你工作里的例子说明为什么。
3. 合成数据为什么直到最近才真正可行？训练前必须做什么？
4. JSONL 和一个大 JSON 文件的区别是什么？训练数据为什么惯用 JSONL？
5. `datasets` 加载本地 JSONL 的写法是什么？加载后怎么看前 3 条、字段类型、样本总数？
6. pandas 读写 xlsx 为什么需要 openpyxl？Excel 到 JSONL 要经过哪几步？
7. instruction、input、output 三元组和 `apply_chat_template` 要的 messages 是什么关系？谁负责转换？

交作业：`.py` 放本目录，发控制台输出（尤其覆盖度报告），说踩坑点。

## 四、待补

- PDF 第八章：数据合成方法细节（程序化合成、模型合成、筛选过滤）
- PDF 第八章：工具使用类数据的样例
- 你的工作领域素材样本（第一批 5 条范例打磨）
