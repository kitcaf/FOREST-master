# 序列到序列扩散预测模型 (FOREST - Seq2Seq)

这个项目实现了一个基于序列到序列 (Seq2Seq) 架构的社交网络信息传播预测模型。给定一个信息传播序列，模型能够预测后续的3个传播节点。

## 主要特点

- **序列到序列架构**：使用编码器-解码器结构，将传播序列映射到预测的后续节点序列
- **增强的用户表示**：结合了意图感知自门控机制和社交网络GCN，学习更丰富的用户表示
- **长短期注意力**：分别捕捉级联序列的长期和短期传播模式
- **时间感知**：模型利用传播时间间隔信息来优化预测

## 数据集

模型使用Twitter数据集，包含：
- 传播扩散序列 (`data/twitter/cascade.txt`)
- 用户社交关系网络 (`data/twitter/edges.txt`)

## 模型结构

1. **用户嵌入增强模块**：
   - 结合了DisenIDP中的意图感知自门控机制
   - 使用MS-HGAT中的社交网络GCN表示
   - 融合基本、兴趣和依赖三种嵌入表示

2. **编码器**：
   - 基于GRU的编码器
   - 结合时间间隔特征增强序列表示

3. **长短期注意力**：
   - 长期影响：基于源节点（第一个节点）的注意力
   - 短期影响：基于最近状态的注意力
   - 将长短期上下文融合用于解码

4. **解码器**：
   - 基于GRU的解码器
   - 利用注意力机制生成预测
   - 支持教师强制训练

## 使用方法

### 安装依赖

```bash
pip install torch numpy scipy scikit-learn
```

### 训练模型

```bash
python main.py -data_name twitter -batch_size 32 -n_epochs 30 -learning_rate 0.001
```

### 参数说明

```
-data_name: 数据集名称，默认为twitter
-split_ratio: 输入序列与目标序列的分割比例，默认为0.7
-batch_size: 批次大小，默认为32
-max_seq_length: 最大序列长度，默认为100
-embed_dim: 嵌入维度，默认为64
-hidden_size: 隐藏层大小，默认为128
-n_layers: GRU层数，默认为2
-dropout: dropout率，默认为0.1
-n_epochs: 训练轮数，默认为30
-learning_rate: 学习率，默认为0.001
-clip: 梯度裁剪阈值，默认为1.0
-teacher_forcing_ratio: 教师强制比例，默认为0.5
-save_path: 模型保存路径，默认为checkpoints/seq2seq_model.pt
```

## 评估指标

- **Hits@k**：预测列表中前k个元素中包含正确预测的比例
- **MAP@k**：平均精度，衡量预测排序的质量

## 参考模型

- **DisenIDP**：使用了自监督分解用户和级联表示增强扩散预测
- **MS-HGAT**：利用超图注意力网络学习用户静态和动态交互

## FOREST
source code for IJCAI 2019 paper "Multi-scale Information Diffusion Prediction with Reinforced Recurrent Networks"

### Description

### Cite

If you find the code useful for your research, please kindly cite this paper:

```
@inproceedings{ijcai2019-560,
  title     = {Multi-scale Information Diffusion Prediction with Reinforced Recurrent Networks},
  author    = {Yang, Cheng and Tang, Jian and Sun, Maosong and Cui, Ganqu and Liu, Zhiyuan},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {4033--4039},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/560},
  url       = {https://doi.org/10.24963/ijcai.2019/560},
}

```
### Contact
albertyang33@gmail.com
