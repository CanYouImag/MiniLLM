# train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 初始化 BPE 分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 训练器配置
trainer = BpeTrainer(
    vocab_size=500,           # 文言文词汇少，500 足够
    min_frequency=1,          # 单字也保留（文言文单字成义）
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 训练过程
tokenizer.train(files=["data/text.txt"], trainer=trainer)

# 保存
tokenizer.save("tokenizer/tokenizer.json")
print("已保存到 tokenizer/tokenizer.json")