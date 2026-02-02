# model.py
import torch
import torch.nn as nn

class ClassicalLanguageModel(nn.Module):
	def __init__(self, vocab_size=600, embed_dim=256, num_heads=8, num_layers=6):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embed_dim)
		self.vocab_size=vocab_size
		self.pos_embed = nn.Embedding(128, embed_dim)  # 最大支持128个字

		decoder_layer = nn.TransformerDecoderLayer(
			d_model=embed_dim,
			nhead=num_heads,
			dim_feedforward=512,
			batch_first=True,
			dropout=0.1
		)
		self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
		self.fc = nn.Linear(embed_dim, vocab_size)
		self.vocab_size = vocab_size
		self.max_len = 64

	def forward(self, idx, targets=None):
		B, T = idx.shape
		pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
		x = self.embed(idx) + self.pos_embed(pos)

		# 因果掩码（不能看未来）
		causal_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(idx.device)
		x = self.transformer(x, memory=x, tgt_mask=causal_mask)
		logits = self.fc(x)

		loss = None
		if targets is not None:
			loss = torch.nn.functional.cross_entropy(
				logits.view(-1, self.vocab_size),
				targets.view(-1)
			)
		return logits, loss

	def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty = 1.0):
		"""生成文言文"""
		self.eval()
		with torch.no_grad():
			for _ in range(max_new_tokens):
				idx_cond = idx[:, -self.max_len:]
				logits, _ = self(idx_cond)
				logits = logits[:, -1, :] / temperature

				# 重复惩罚
				if repetition_penalty != 1.0:

					# 创建一个与logits同型的mask
					penalty_mask = torch.zeros_like(logits,dtype=torch.bool)

					# 对每个样本，标记已出现的token
					for i in range(logits.shape[0]):
						generated_tokens = idx[i].tolist()
						penalty_mask[i, generated_tokens] = True

					# 应用惩罚：logits = logits / penalty（>1 则降低）
					logits = torch.where(
						penalty_mask,
						logits / repetition_penalty,
						logits
					)

				# top_k检测
				if top_k is not None:
					values, indices = torch.topk(logits, top_k)
					min_values = values[:, -1].unsqueeze(-1).expand_as(logits)
					logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

				probs = torch.softmax(logits, dim=-1)
				idx_next = torch.multinomial(probs, num_samples=1)
				idx = torch.cat([idx, idx_next], dim=1)
		return idx
