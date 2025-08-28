# generate.py
import torch
from model import ClassicalLanguageModel
from tokenizers import Tokenizer

# 加载分词器和模型
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()     # 获取vocab_size的真实大小

print(f"分词器词汇表大小：{vocab_size}")
# 传递真实情况下的vocab_size大小
model = ClassicalLanguageModel(vocab_size = vocab_size)
model.load_state_dict(torch.load("classical_model.pth", map_location='cpu', weights_only = True))
model.eval()

def generate_text(prompt, max_new_tokens=64, temperature=0.9, top_k=100, repetition_penalty = 1.2):
    # 编码输入
    encoded = tokenizer.encode(prompt)
    ids = torch.tensor([encoded.ids])

    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k, repetition_penalty = repetition_penalty)

    # 解码输出
    text = tokenizer.decode(output_ids[0].tolist())
    return text

# 测试生成
prompts = [
    "子卿足下",
    "呜呼",
    "且夫",
    "陵与子卿",
    "自从初降",
    "身之"
]

for p in prompts:
    print(f"【输入】: {p}")
    print(f"【生成】: {generate_text(p, temperature = 0.9, top_k = 100, repetition_penalty = 1.2)}\n")