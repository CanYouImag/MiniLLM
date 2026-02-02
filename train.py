# train.py
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import ClassicalLanguageModel
from tqdm import tqdm

# 加载分词器
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# 数据集类
class ClassicalDataset(Dataset):
	def __init__(self, file_path, seq_len=32):
		with open(file_path, 'r', encoding='utf-8') as f:
			text = f.read().replace('\n', '')  # 去掉换行符
		self.tokens = tokenizer.encode(text).ids
		self.seq_len = seq_len

	def __len__(self):
		return max(1, len(self.tokens) - self.seq_len)

	def __getitem__(self, idx):
		start = idx
		end = idx + self.seq_len + 1
		chunk = self.tokens[start:end]
		x = torch.tensor(chunk[:-1])
		y = torch.tensor(chunk[1:])
		return x, y

# 创建数据集和数据加载器
dataset = ClassicalDataset("data/text.txt", seq_len=96)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 获取真实词汇表大小
vocab_size = tokenizer.get_vocab_size()
print(f"分词器词汇表大小: {vocab_size}")

# 创建模型
model = ClassicalLanguageModel(vocab_size=vocab_size, embed_dim=256, num_heads=8, num_layers=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)   # 学习率调度器

# 训练
model.train()
best_loss = float('inf')
patience = 0
for epoch in range(700):
	total_loss = 0

	pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", leave=(epoch % 50 == 0))
	for x, y in pbar:
		x, y = x.to(device), y.to(device)
		optimizer.zero_grad()
		_, loss = model(x, y)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		total_loss += loss.item()
		current_avg_loss = total_loss / (pbar.n + 1)  # 当前 epoch 的平均 loss

		# 实时更新进度条右侧信息
		current_lr = optimizer.param_groups[0]['lr']
		pbar.set_postfix({
			'loss': f"{loss.item():.4f}",
			'avg_loss': f"{current_avg_loss:.4f}",
			'lr': f"{current_lr:.2e}"
		})

	scheduler.step()    # 更新学习率

	if epoch % 50 == 0:
		avg_loss = total_loss / len(dataloader)
		print()			# 清空tqdm残留
		print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.4f}")

		# 早停机制
		if avg_loss < best_loss:
			best_loss = avg_loss
			patience = 0
			torch.save(model.state_dict(), "classical_model.pth") # 保存模型
		else:
			patience += 1   # 增加耐心
			if patience > 50:   # 10个epoch还没有改善就停止训练
				print(" patience > 50, 训练结束")
				break

# 保存模型
torch.save(model.state_dict(), "classical_model.pth")
print("✅ 文言文模型已保存到 classical_model.pth")