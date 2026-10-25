# P1 Dataset

## 1. dataset
dataset是一个数据集的意思,通过 load_dataset 下载/读取/加载数据集.

## 2. Tensor

[video](https://www.bilibili.com/video/BV19C411G7Zw/?spm_id_from=333.337.search-card.all.click&vd_source=b5c04f54b8a7ce0b4d5deef9989f7f9f)

tensor 是一种数据结构，这种数据结构就是无限多维数组。

并且它可以直接交给 GPU 进行计算。

不管是文件，还是图片，还是任何文件，都可以转成 tensor 结构。

tensor 是被设计用来利用 GPU 加速计算的。

---

tensor 是存储输入数据的方式，


```python
import torch
import pandas as pd

print("--- 1. 处理 CSV 表格数据 ---")
# 假装我们有一个 csv 文件，数据是：年龄, 工资
# 在 Python 里用 pandas 直接建一个假表格
df = pd.DataFrame({
    'age': [20, 30, 40],
    'salary': [5000.0, 10000.0, 15000.0]
})
# 直接把这个表格转成 Tensor！看，就这么简单！
csv_tensor = torch.tensor(df.values, dtype=torch.float32)
print("CSV 变成的 Tensor:\n", csv_tensor)
print("形状:", csv_tensor.shape) # 输出 torch.Size([3, 2]) 3行2列

print("\n--- 2. 处理 TXT 文本数据 ---")
# 假装我们用分词器把文字变成了数字 ID
text_ids = [101, 202, 303, 404] # 代表 "我", "爱", "学", "习"
# 把数字ID转成 Tensor
text_tensor = torch.tensor(text_ids)
print("文本变成的 Tensor:\n", text_tensor)
print("形状:", text_tensor.shape) # 输出 torch.Size([4])
```


## Transforms的使用

是什么：

草，说人话就是：**`transforms` 就是个“图片加工厂”，专门用来把原始图片“盘”成模型能吃的形状。**

你之前写 `Dataset` 的时候，用 `Image.open()` 读出来的是个啥？是一张原汁原味的 **PIL 图片对象**。尺寸乱七八糟，格式还是 JPG 像素。
但模型（`nn.Module`）是个极度挑食的巨婴，它只吃**固定尺寸（比如 224x224）、数据格式为 0.0~1.0 浮点数的 `Tensor`（张量）**。

`transforms` 就是干这个脏活累活的。结合你的 Java 思维，它相当于你 Spring 框架里的 **`Filter` 过滤器**，或者 **`ImageUtils` 工具类**。

### 🛠️ `transforms` 里面最核心的 3 个工具（API）：

1.  **`transforms.ToTensor()`（绝对核心）**：
    *   **作用**：把 `PIL` 图片转成 `Tensor` 张量。同时把原本 0~255 的整数像素，除以 255，变成 0.0~1.0 的浮点数。
    *   **Java 比喻**：相当于把 `String` 类型的 JSON 变成了 `Object` 实体类。
2.  **`transforms.Resize((224, 224))`**：
    *   **作用**：不管你原图是 1000x800 还是 500x300，统统裁剪/缩放到 224x224。
    *   **为什么**：因为 `DataLoader` 打包成 Batch 的时候，必须要求所有图片的尺寸完全一致，不然没法拼成矩阵！
3.  **`transforms.Compose([...])`**：
    *   **作用**：流水线组装。把上面几个步骤串起来，一步到位。
    *   **Java 比喻**：相当于 Java 里的 `Stream().filter().map().collect()` 链式调用。

### 🚀 看看代码里怎么用（套路模板）：

```python
from torchvision import transforms

# 这就是你的加工流水线
# 第一步：把图片尺寸变成 224x224
# 第二步：转成 Tensor
my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 以后你在 Dataset 的 __init__ 里把它传进去
# 然后在 __getitem__ 里这么用：
# img = my_transform(img)
```

---

### 🔄 串联你脑子里的流水线：

你想想，你的数据是怎么一点点变成矩阵，送进显卡的？
1.  **`Dataset`** 拿到图片路径，`Image.open()` 读出原始图片（还没法吃）。
2.  **`transforms`** 一顿爆改，变成 224x224 的 `Tensor`（可以吃了）。
3.  **`DataLoader`** 把 4 张 `Tensor` 叠在一起，变成一个形状为 `(4, 3, 224, 224)` 的大矩阵（打包成 Batch）。
4.  送进 **模型 (`nn.Module`)**。

**没有 transforms，DataLoader 根本没东西可以打包，模型只能吃土。**

这下懂了吧？进度条拉到讲 `transforms` 的地方，把 `ToTensor` 和 `Compose` 这两玩意儿看懂，然后立刻去找 `DataLoader` 的视频！



