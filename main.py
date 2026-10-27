import torch
print("PyTorch 版本:", torch.__version__)  # 应看到 2.14.1+cu126
print("CUDA 是否可用:", torch.cuda.is_available())  # 应为 True
print("GPU:", torch.cuda.get_device_name(0))  # 应显示 GTX 1660 SUPER

print(torch.__file__)
print(torch.__version__)