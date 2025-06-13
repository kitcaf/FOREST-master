# 图增强序列到序列模型 (Graph-Augmented Seq2Seq)

这个项目实现了一个图增强序列到序列模型，用于社交网络信息扩散预测任务。该模型结合了序列建模和社交网络结构信息，以提高预测准确性。

## 特点

- **图增强序列模型**：利用社交网络图结构信息增强序列预测
- **社交修正机制**：在解码阶段使用社交网络结构进行预测修正
- **多种评估指标**：支持Hits@k和MAP@k等多种评估指标
- **灵活的训练参数**：支持多种优化器、学习率调度器和训练策略

## 文件结构

- `main.py`: 主程序，包含训练、评估和预测功能
- `seq2seq_model.py`: 图增强序列到序列模型实现
- `seq2seq_dataloader.py`: 数据加载和预处理
- `metrics.py`: 评估指标实现
- `Constants.py`: 常量定义
- `Optim.py`: 优化器和学习率调度器

## 数据格式

数据应放在`data/{数据集名称}/`目录下，包含以下文件：

- `cascadetrain.txt`: 训练集级联数据
- `cascadevalid.txt`: 验证集级联数据
- `cascadetest.txt`: 测试集级联数据
- `edges.txt`: 社交网络边信息
- `dw{embed_dim}.txt`: 预训练节点嵌入（可选）

## 使用方法

### 训练模型

```bash
python main.py --data twitter --epochs 20 --lr 0.001 --use_social_graph --save_model
```

### 评估模型

```bash
python main.py --data twitter --load_model models/twitter_model_final.pt --cuda
```

### 生成预测

```bash
python main.py --data twitter --load_model models/twitter_model_final.pt --predict --beam_search --social_correction --cuda
```

## 参数说明

### 数据参数
- `--data`: 数据集名称，默认为'twitter'
- `--split_ratio`: 训练集比例，默认为0.8
- `--batch_size`: 批次大小，默认为32

### 模型参数
- `--embed_dim`: 嵌入维度，默认为128
- `--hidden_size`: 隐藏层大小，默认为256
- `--n_layers`: 层数，默认为2
- `--dropout`: Dropout比例，默认为0.1
- `--use_social_graph`: 是否使用社交图，默认为False

### 训练参数
- `--epochs`: 训练轮数，默认为20
- `--lr`: 学习率，默认为0.001
- `--optimizer`: 优化器类型（adam, sgd, rmsprop），默认为'adam'
- `--scheduler`: 学习率调度器类型（step, plateau），默认为'step'
- `--cuda`: 是否使用CUDA，默认为False
- `--seed`: 随机种子，默认为42
- `--save_model`: 是否保存模型，默认为False
- `--load_model`: 加载预训练模型路径

### 预测参数
- `--predict`: 是否进行预测，默认为False
- `--beam_search`: 是否使用束搜索，默认为False
- `--beam_size`: 束大小，默认为5
- `--social_correction`: 是否使用社交网络修正，默认为False
- `--temperature`: 温度参数，默认为1.0

## 输出文件

所有输出文件将保存在`output/`目录下：

- `{数据集名称}_training_history.json`: 训练历史
- `{数据集名称}_eval_metrics.json`: 验证集评估指标
- `{数据集名称}_test_metrics.json`: 测试集评估指标
- `{数据集名称}_predictions.txt`: 预测结果

模型文件将保存在`models/`目录下：

- `{数据集名称}_model_epoch{轮数}.pt`: 中间模型
- `{数据集名称}_model_final.pt`: 最终模型

## 评估指标

- **Hits@k**: 真实标签是否在预测的前k个结果中
- **MAP@k**: 平均精度@k，考虑预测结果的排序质量

这些指标会针对预测序列的前3个位置分别计算，并取平均值。








