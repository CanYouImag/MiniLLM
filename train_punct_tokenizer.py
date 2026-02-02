# train_punct_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

# 直接按字符分词（自动保留所有标点）
def char_iterator(file_path):
	with open(file_path, "r", encoding="utf-8") as f:
		text = f.read()
		# 移除换行符，但保留所有标点：，。；：？！“”‘’（）等
		text = text.replace("\n", " ")
		for char in text:
			yield char

# 初始化 Unigram 分词器（适合小词汇表）
tokenizer = Tokenizer(Unigram())

trainer = UnigramTrainer(
	vocab_size=2000,  # 文言文+标点
	special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
	unk_token="[UNK]"
)

# 训练：每个字符作为一个单元
tokenizer.train_from_iterator(char_iterator("data/text.txt"), trainer=trainer)

# 保存
tokenizer.save("tokenizer/tokenizer.json")
print("✅ 带标点的字符级分词器已生成！")
print("词汇表大小:", tokenizer.get_vocab_size())