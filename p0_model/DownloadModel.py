import os
# 必须在导入 transformers 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"正在加载模型 {model_name}...")

print("1. 开始加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("2. Tokenizer 加载完毕！")

# 主流做法：保留 auto 自动分流，但显式限制 GPU 显存预算
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={0: "4GB", "cpu": "2GB"},  # 【死命令】总共给8G系统留2G，显卡给4G，CPU只准用2G！
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,  # 【保命神器】这个必加，它是一边从硬盘读一边往显卡塞，不占内存
)

model.eval()  # 推理前切 eval 模式，这也是业界标配
messages = [
    {"role": "system", "content": "你是一个暴躁但技术极其牛逼的AI架构师。"},
    {"role": "user", "content": "用一句话告诉我，怎么判断我的模型有没有跑在显卡上？"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("\n模型正在疯狂计算中...\n")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=100,
    temperature=0.7
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("====== AI 回复 ======")
print(response)