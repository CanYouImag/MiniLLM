# check_gpu.py
import torch

print("=" * 50)
print("✅ PyTorch 版本:", torch.__version__)
print("✅ CUDA 可用:", torch.cuda.is_available())
print("✅ CUDA 版本:", torch.version.cuda)
print("✅ cuDNN 版本:", torch.backends.cudnn.version())
print("✅ GPU 数量:", torch.cuda.device_count())

if torch.cuda.is_available():
	print("✅ 当前 GPU:", torch.cuda.get_device_name(0))
	# 测试 GPU 计算
	x = torch.randn(1000, 1000).cuda()
	y = torch.randn(1000, 1000).cuda()
	z = torch.mm(x, y)
	print("✅ GPU 矩阵乘法测试通过！")
else:
	print("❌ 警告：CUDA 不可用！你可能装的是 CPU 版本。")

print("=" * 50)