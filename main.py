import pandas as pd
import os

# 模拟真实业务里的脏数据
data = {
    # 把范围改成 1到106，正好生成 105 个
    "user_id": [f"u{i}" for i in range(1, 106)],
    "comment_text": [
        "质量很好，下次还来买！",
        "",  # 坑1：空评论
        "垃圾东西，刚用一天就坏了，差评！",
        "发货速度很快，客服态度也不错，就是包装有点破损。",
        "😂😂😂😂😂😂😂" * 20,  # 坑2：超长且全是表情包
        None, # 坑3：None值
        "这是一条很长很长很长的评论，" * 50 # 坑4：极长文本
    ] * 15, # 凑够 105 条
    "star_rating": [5, 1, 4, 3, 2, None, 5] * 15 # 凑够 105 条
}

# 确保你要存数据的文件夹存在
os.makedirs("./p1_dataloader", exist_ok=True)

df = pd.DataFrame(data)
df.to_csv("./p1_dataloader/comments.csv", index=False, encoding="utf-8")
print("脏数据生成完毕！去拿你的 Dataset 清洗它吧！")