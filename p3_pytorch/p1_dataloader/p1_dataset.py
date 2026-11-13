from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download

# 通过 transforms.ToTensor 解决2个问题:
img_path = "my_datasets/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

# 1. transforms 该如何使用

# ToTensor
tensor_img = transforms.ToTensor()(img)
print(tensor_img)


# Normalize 归一化，数据标准化，数据分布过于极端会导致的梯度下降
print(tensor_img[0][0][0]) # 标准化之前第一个像书店数据
# 它接收两个列表：第一个是均值（mean），第二个是标准差（std）
img_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor_img)
print(img_norm[0][0][0]) # 标准化之后像素点参数，其实标准化后的图，人眼看起来会非常诡异、灰暗，因为数据被拉扯过了


# Resize 将图片裁减成指定的规格，因为dataloader打包的时候必须要求图片的尺寸完全一致
print(img.size)
# PIL --> Resize --> Tensor
img_resize = transforms.Resize((224, 224))
img_resize = transforms.ToTensor()(img_resize)
print(img_resize)


# Compose
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img = transform(img)
print(img.shape)


###################################################################################
###################################################################################


# 读取数据集
# 0. 数据准备
# ds = load_dataset("antsanchez/synthetic-fishing-products", split="train", cache_dir="./my_datasets")
# df = ds.to_pandas()
# df.to_csv("./my_datasets/my_local_fishing.csv", index=False)
# ds.to_json("./my_datasets/fishing_data.json", force_ascii=False)


# 下载整个数据集仓库到本地目录
snapshot_download(
    repo_id="Boyoungc/pathomics",  # 数据集仓库ID
    repo_type="dataset",  # 必须指定是 dataset
    local_dir="./my_datasets",  # 本地保存路径
    allow_patterns="*",  # 下载全部文件
)



# 1. 加载本地标准 huggingFase 数据集
# ds = load_dataset("./my_datasets/YiMeng-SYSU___chinese-logic-sentiment-dataset", split="train")
# print(ds[0])

# 2. 记载本地 CSV 文件
# ds = load_dataset("csv", data_files="./my_datasets/*.csv", split="train")
# print(ds[0])

# 3. 加载 json 文件
# ds = load_dataset("json", data_files="./my_datasets/*.json", split="train")
# print(ds[0])

# 4. 加载图片文件
ds_images = load_dataset("imagefolder", data_dir="./my_datasets/hymenoptera_data/train", split="train")
print(ds_images[0])

# 2. 定义 transforms 装载图片格式的流水线
compose = transforms.Compose([
    # 新增这一行：强制把不管什么通道的 PIL 图片，全他妈转成 RGB (3通道)
    transforms.Lambda(lambda img: img.convert("RGB")),
    # 裁剪
    transforms.Resize((224, 224)),
    # 转为 Tensor 数据格式
    transforms.ToTensor(),
    # 归一化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 3. 将 transforms 应用到图片上
def apply_transforms(examples):
    # 注意：HuggingFace 的 image 列可能是 PIL 对象，直接过 compose 即可
    examples["image"] = [compose(img) for img in examples["image"]]
    return examples


# 4. 批量处理
ds_images = ds_images.map(apply_transforms, batched=True)

# 5. 把数据格式设置成 PyTorch 认识的 Tensor
ds_images.set_format(type="torch", columns=["image", "label"])

# 6. 放进 DataLoader 打包
dataloader = DataLoader(ds_images, batch_size=4, shuffle=True)

# 7. 见证奇迹！
for batch in dataloader:
    print("图片 Batch 形状:", batch["image"].shape)  # 应该是 [4, 3, 224, 224]
    print("标签 Batch 形状:", batch["label"].shape)  # 应该是 [4]
    break
