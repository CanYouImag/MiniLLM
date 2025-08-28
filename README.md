# MiniLLM - 古文语言模型项目

## 项目简介

MiniLLM是一个基于PyTorch实现的轻量级中文古文（文言文）语言模型项目。该项目旨在构建一个能够理解和生成文言文的小型Transformer模型，可用于古文文本生成、补全等任务。

## 功能特性

- 基于Transformer架构的解码器-only模型
- 支持文言文文本生成
- 使用BPE分词器处理中文文本
- 实现了多种文本生成策略（温度采样、top-k采样、重复惩罚等）
- 轻量级设计，适合学习和实验

## 项目结构

```
MiniLLM/
├── data/
│   └── text.txt          # 训练数据文件
├── tokenizer/
│   └── tokenizer.json    # 分词器配置文件
├── model.py              # 模型定义
├── train_tokenizer.py    # 分词器训练脚本
├── train.py              # 模型训练脚本
├── generate.py           # 文本生成脚本
└── classical_model.pth   # 训练好的模型权重文件
```


## 环境依赖

### Python版本
- Python 3.7+

### 核心依赖库
- torch>=1.9.0
- tokenizers>=0.10.0

### 安装依赖
```bash
pip install torch tokenizers
```


## 使用指南

### 1. 准备训练数据
在`data/text.txt`中放入用于训练的文言文文本数据，确保文件编码为UTF-8。

### 2. 训练分词器
```bash
python train_tokenizer.py
```


### 3. 训练模型
```bash
python train.py
```


### 4. 生成文本
```bash
python generate.py
```


## 配置说明

### 模型配置
在[model.py](file://C:\Users\11831\PycharmProjects\MiniLLM\model.py)中可以调整以下参数：
- `embed_dim`: 词嵌入维度（默认256）
- `num_heads`: 注意力头数（默认8）
- `num_layers`: Transformer层数（默认6）
- `dim_feedforward`: 前馈网络维度（默认512）

### 训练配置
在[train.py](file://C:\Users\11831\PycharmProjects\MiniLLM\train.py)中可以调整以下参数：
- [seq_len](file://C:\Users\11831\PycharmProjects\MiniLLM\train.py#L0-L0): 序列长度（默认64）
- `batch_size`: 批次大小（默认16）
- `learning_rate`: 学习率（默认3e-4）
- `epochs`: 训练轮数（默认1000）

### 生成配置
在[generate.py](file://C:\Users\11831\PycharmProjects\MiniLLM\generate.py)中可以调整以下参数：
- `temperature`: 温度参数（控制随机性，默认0.9）
- `top_k`: Top-K采样参数（默认100）
- `repetition_penalty`: 重复惩罚系数（默认1.2）

## 训练建议

1. **数据质量**：确保训练数据为高质量的文言文文本
2. **训练时间**：根据硬件配置，训练可能需要数小时
3. **超参数调优**：可根据实际效果调整模型大小和训练参数
4. **早停机制**：训练中实现了早停机制防止过拟合

## 注意事项

1. 项目默认使用CPU训练，如有GPU可自动切换
2. 生成文本质量与训练数据量和质量密切相关
3. 模型大小会影响训练时间和生成质量的平衡
4. 确保所有文本文件使用UTF-8编码

## 安全说明

项目使用`torch.load`时设置了`weights_only=True`参数，防止加载恶意模型文件时执行任意代码。

## 许可证

GPL-3.0。

---

**提示**：首次使用时请按顺序执行[train_tokenizer.py](file://~\train_tokenizer.py)、[train.py](file://~\train.py)、[generate.py](file://~\generate.py)三个脚本。
