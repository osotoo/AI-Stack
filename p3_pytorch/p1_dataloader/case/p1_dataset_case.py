"""
练习一：电商评论情感分析（NLP核心：处理变长文本）
业务背景：公司爬了一堆京东的商品评论数据，存在一个 comments.csv 文件里（字段：user_id, comment_text, star_rating）。你要把它做成能喂给模型的 Dataset。

技术要求：

数据清洗：如果 comment_text 是空的，或者 star_rating 缺失，直接丢弃这条数据。

自定义 Dataset：写一个 ReviewDataset 类继承 Dataset。
在 __getitem__ 里，使用 transformers 的 AutoTokenizer（比如 bert-base-chinese），把文本转成 input_ids。
如果评价分数（star_rating）大于等于4，标签设为1（好评），小于等于2设为0（差评），中间的3丢掉。

核心考点（必做）：文本有长有短。如果你直接 DataLoader(batch_size=4)，绝对会报错（形状不一致无法打包）。你需要写一个自定义的 collate_fn，在打包时，动态计算当前 Batch 里最长的那句话，把其他短句补齐（Padding）。
提示：这是企业里做文本分类、甚至微调大模型前必须掌握的基石。
"""
from datasets import load_dataset

# 1.读取文件
ds = load_dataset("csv", data_files="comments.csv", split="train")
print(ds[0])

# 2. 数据清洗：如果 comment_text 是空的，或者 star_rating 缺失，直接丢弃这条数据。
