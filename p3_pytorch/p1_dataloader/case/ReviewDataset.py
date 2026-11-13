import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer



class ReviewDataset(Dataset):

    def __init__(self, csv_file):
        # 1. 读取数据
        df = pd.read_csv(csv_file)

        # 2. 数据清洗
        # 删掉文本为空的行
        df = df.dropna(subset=["comment_text"])
        df = df[df["comment_text"].str.strip() != ""]

        # 删掉评分为空或者等于3的行
        df = df.dropna(subset=["star_rating"])
        df = df[df["star_rating"] != 3]

        # 3. 转换标签 (4,5 --> 1, 1,2 --> 0 )
        # 用apply 加 lambda 快速转换
        df["label"] = df["star_rating"].apply(lambda x: 1 if x >= 4 else 0)

        # 4. 重置索引(因为删了行, 索引会断,必须重置)
        df = df.reset_index(drop=True)

        # 把清洗好的数据村委类属性
        self.texts = df["comment_text"].tolist()
        self.labels = df["label"].tolist()

        # 5. 初始化 Tokenizer
        AutoTokenizer.from_pretrained("bert-base-chinese")


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # 1. 拿到文本和标签
        text = self.texts[index]
        label = self.labels[index]

        # 2. 文本赚 Tensor


        return


