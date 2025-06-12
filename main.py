"""
Graph-Augmented Seq2Seq模型训练和评估主程序
用于社交网络信息扩散预测任务
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from seq2seq_dataloader import Seq2SeqDataLoader
from seq2seq_model import GraphAugmentedSeq2Seq
from Optim import ScheduledOptim
from metrics import portfolio
import Constants

# 设置随机种子
def set_seed(seed):
    """设置随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 定义参数
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="图增强序列到序列模型用于社交网络信息扩散预测")
    
    # 数据参数
    parser.add_argument('-data_name', type=str, default='twitter', help='数据集名称')
    parser.add_argument('-split_ratio', type=float, default=0.7, help='输入序列与目标序列的分割比例')
    parser.add_argument('-batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('-max_seq_length', type=int, default=100, help='最大序列长度')
    
    # 模型参数
    parser.add_argument('-embed_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('-hidden_size', type=int, default=256, help='隐藏层大小')
    parser.add_argument('-n_layers', type=int, default=2, help='GRU层数')
    parser.add_argument('-dropout', type=float, default=0.2, help='dropout率')
    parser.add_argument('-bidirectional', type=bool, default=True, help='是否使用双向编码器')
    
    # 图神经网络参数
    parser.add_argument('-use_hypergraph', type=bool, default=True, help='是否使用超图特征')
    parser.add_argument('-use_social_constraint', type=bool, default=True, help='是否使用社交约束')
    parser.add_argument('-disable_all_constraints', action='store_true', help='禁用所有社交约束和图特征，纯序列模型')
    parser.add_argument('-window_size', type=int, default=5, help='超图滑动窗口大小')
    
    # 训练参数
    parser.add_argument('-n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('-learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('-clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('-teacher_forcing_ratio', type=float, default=0.5, help='教师强制比例')
    parser.add_argument('-n_warmup_steps', type=int, default=4000, help='预热步数')
    parser.add_argument('-weight_decay', type=float, default=1e-5, help='权重衰减（L2正则化）')
    
    # 其他参数
    parser.add_argument('-no_cuda', action='store_true', help='不使用CUDA')
    parser.add_argument('-seed', type=int, default=42, help='随机种子')
    parser.add_argument('-save_path', type=str, default='checkpoints/graph_seq2seq_model.pt', help='模型保存路径')
    parser.add_argument('-log_interval', type=int, default=10, help='日志间隔')
    parser.add_argument('-patience', type=int, default=10, help='早停耐心值')
    
    args = parser.parse_args()
    
    # 设置CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # 如果选择禁用所有约束，则设置相关参数
    if args.disable_all_constraints:
        args.use_hypergraph = False
        args.use_social_constraint = False
        print("警告: 已禁用所有社交网络约束和图特征，将使用纯序列模型")
    
    return args

# 训练一个epoch
def train_epoch(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, log_interval, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    n_batches = 0
    
    # 计算总批次数
    total_batches = len(data_loader.get_train_batches())
    print(f"\n开始训练第 {epoch} 轮 | 共 {total_batches} 批次")
    print("="*50)
    
    for batch_idx, batch in enumerate(data_loader.get_train_batches()):
        # 计算进度百分比
        progress = (batch_idx + 1) / total_batches * 100
        
        # 获取数据
        src = batch['src']
        tgt = batch['tgt']
        src_lengths = batch['src_lengths']
        time_intervals = batch['time_intervals']
        
        # 前向传播
        outputs = model(src, src_lengths, time_intervals, tgt, teacher_forcing_ratio)
        
        # 计算损失 - 采用递减权重方案
        loss = 0
        position_weights = [1.2, 1.0, 0.8]  # 对前三个预测位置的权重，靠前的位置权重更大
        
        for t in range(min(tgt.size(1) - 1, 3)):  # 只关注前3个预测位置
            pos_loss = criterion(outputs[:, t, :], tgt[:, t+1])
            loss += position_weights[t] * pos_loss
        
        # 除以总权重而非位置数量
        loss = loss / sum(position_weights[:min(tgt.size(1) - 1, 3)])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        clip_grad_norm_(model.parameters(), clip)
        
        # 参数更新
        optimizer.step()
        
        # 更新学习率
        lr = optimizer.update_learning_rate()
        
        total_loss += loss.item()
        n_batches += 1
        
        # 打印日志
        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / max(1, batch_idx % log_interval + 1)
            
            # 计算预计剩余时间
            if batch_idx > 0:
                time_per_batch = elapsed / (batch_idx % log_interval + 1)
                remaining_batches = total_batches - (batch_idx + 1)
                eta_seconds = remaining_batches * time_per_batch
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            else:
                eta_str = "计算中..."
            
            # 进度条
            bar_length = 30
            filled_length = int(bar_length * (batch_idx + 1) // total_batches)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"\r[{bar}] {progress:.1f}% | 批次: {batch_idx+1}/{total_batches} | "
                  f"损失: {avg_loss:.4f} | 学习率: {lr:.6f} | 预计剩余: {eta_str}", end='')
            
            if (batch_idx % log_interval == 0 and batch_idx > 0) or batch_idx == total_batches - 1:
                print()  # 换行
                total_loss = 0
                start_time = time.time()
    
    print(f"\n第 {epoch} 轮训练完成")
    return total_loss / n_batches if n_batches > 0 else float('inf')

# 评估
def evaluate(model, data_loader, criterion, split='valid', k_list=[10, 50, 100]):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    total_scores = {f'hits@{k}': 0.0 for k in k_list}
    total_scores.update({f'map@{k}': 0.0 for k in k_list})
    total_len = 0
    n_batches = 0
    
    with torch.no_grad():
        if split == 'valid':
            data_batches = data_loader.get_valid_batches()
        else:  # test
            data_batches = data_loader.get_test_batches()
        
        for batch in data_batches:
            # 获取数据
            src = batch['src']
            tgt = batch['tgt']
            src_lengths = batch['src_lengths']
            time_intervals = batch['time_intervals']
            
            # 前向传播
            outputs = model(src, src_lengths, time_intervals, tgt, 0.0)  # 不使用教师强制
            
            # 计算损失 - 与训练时一致使用递减权重
            loss = 0
            position_weights = [1.2, 1.0, 0.8]
            
            for t in range(min(tgt.size(1) - 1, 3)):  # 只考虑前3个预测
                pos_loss = criterion(outputs[:, t, :], tgt[:, t+1])
                loss += position_weights[t] * pos_loss
            
            # 除以总权重
            loss = loss / sum(position_weights[:min(tgt.size(1) - 1, 3)])
            
            total_loss += loss.item()
            n_batches += 1
            
            # 计算评估指标
            for t in range(min(tgt.size(1) - 1, 3)):  # 只考虑前3个预测
                # 注意：outputs和tgt可能是CUDA张量，portfolio函数中会处理
                scores, scores_len = portfolio(outputs[:, t, :], tgt[:, t+1], k_list=k_list)
                for k in k_list:
                    total_scores[f'hits@{k}'] += scores[f'hits@{k}'] * scores_len
                    total_scores[f'map@{k}'] += scores[f'map@{k}'] * scores_len
                total_len += scores_len
    
    # 计算平均值
    avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
    
    # 计算平均指标
    if total_len > 0:
        for k in k_list:
            total_scores[f'hits@{k}'] /= total_len
            total_scores[f'map@{k}'] /= total_len
    
    return avg_loss, total_scores

# 预测
def predict(model, data_loader, split='test', top_k=3):
    """使用模型进行预测"""
    model.eval()
    predictions = []
    ground_truth = []
    attention_weights = []
    
    with torch.no_grad():
        if split == 'valid':
            data_batches = data_loader.get_valid_batches()
        else:  # test
            data_batches = data_loader.get_test_batches()
        
        for batch in data_batches:
            # 获取数据
            src = batch['src']
            tgt = batch['tgt']
            src_lengths = batch['src_lengths']
            time_intervals = batch['time_intervals']
            
            # 预测
            pred, attn = model.predict(src, src_lengths, time_intervals, max_length=3)
            
            # 将CUDA张量移到CPU
            if pred.is_cuda:
                pred = pred.cpu()
            if tgt.is_cuda:
                tgt = tgt.cpu()
            
            # 收集预测和真实值
            for i in range(pred.size(0)):
                pred_seq = pred[i].tolist()
                true_seq = tgt[i, 1:4].tolist()  # 跳过BOS，取3个真实节点
                
                predictions.append(pred_seq)
                ground_truth.append(true_seq)
                
                # 收集注意力权重（可选）
                batch_attn = [a[i].cpu().numpy() for a in attn]
                attention_weights.append(batch_attn)
    
    return predictions, ground_truth, attention_weights

# 保存预测结果
def save_predictions(predictions, ground_truth, idx2u, file_path):
    """保存预测结果到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("预测序列\t真实序列\n")
        for pred, true in zip(predictions, ground_truth):
            # 将ID转换为用户名
            pred_users = [idx2u[p] if p < len(idx2u) else '<unk>' for p in pred]
            true_users = [idx2u[t] if t < len(idx2u) and t != Constants.PAD else '<pad>' for t in true]
            
            # 写入文件
            f.write(f"{' -> '.join(pred_users)}\t{' -> '.join(true_users)}\n")

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存模型和结果的目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    result_dir = os.path.dirname(args.save_path)
    
    # 创建指标保存文件
    metrics_file = os.path.join(result_dir, f'metrics_{args.data_name}.txt')
    with open(metrics_file, 'w') as f:
        f.write("epoch,train_loss,valid_loss,hits@10,hits@50,hits@100,map@10,map@50,map@100\n")
    
    # 加载数据
    print(f"正在加载 {args.data_name} 数据集...")
    data_loader = Seq2SeqDataLoader(
        data_name=args.data_name,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        cuda=args.cuda,
        shuffle=True,
        loadNE=not args.disable_all_constraints,  # 如果禁用约束，不加载网络嵌入
        max_seq_length=args.max_seq_length
    )
    
    # 确认数据集大小
    print(f"训练集大小: {len(data_loader.train_cascades)} 个级联序列")
    print(f"验证集大小: {len(data_loader.valid_cascades)} 个级联序列")
    print(f"测试集大小: {len(data_loader.test_cascades)} 个级联序列")
    
    # 创建模型
    has_social_data = hasattr(data_loader, 'adj_tensor') and data_loader.adj_tensor is not None
    has_pretrained_embeds = hasattr(data_loader, 'embeds') and data_loader.embeds is not None
    has_hypergraph = hasattr(data_loader, 'HG_User') and data_loader.HG_User is not None and args.use_hypergraph
    
    # 邻接矩阵或超图
    adj_tensor = None
    if not args.disable_all_constraints:
        if has_social_data:
            print("使用社交网络关系数据进行模型训练")
            if has_hypergraph and args.use_hypergraph:
                print("使用超图特征增强社交网络建模")
                adj_tensor = data_loader.HG_User
            else:
                print("使用普通社交网络特征")
                adj_tensor = data_loader.adj_tensor
        else:
            print("警告: 无社交网络关系数据，模型将只使用扩散序列信息")
            
        if has_pretrained_embeds:
            print("使用预训练的用户嵌入")
        else:
            print("警告: 无预训练用户嵌入，将随机初始化用户嵌入")
    else:
        print("已禁用所有社交网络特征，使用纯序列模型")
    
    print(f"创建Graph-Augmented Seq2Seq模型...")
    model = GraphAugmentedSeq2Seq(
        user_size=data_loader.user_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        adj_tensor=adj_tensor if args.use_social_constraint and not args.disable_all_constraints else None,
        pretrained_embeds=data_loader.embeds if has_pretrained_embeds and not args.disable_all_constraints else None,
        bidirectional_encoder=args.bidirectional
    )
    
    if args.cuda:
        print("使用CUDA加速训练")
        model = model.cuda()
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=Constants.PAD)
    
    # 定义优化器
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
        args.hidden_size,
        args.n_warmup_steps
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 训练模型
    best_valid_loss = float('inf')
    best_epoch = 0
    best_metrics = {}
    patience = args.patience
    patience_counter = 0
    
    print("开始训练...")
    for epoch in range(1, args.n_epochs + 1):
        # 训练
        train_loss = train_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip=args.clip,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            log_interval=args.log_interval,
            epoch=epoch
        )
        
        # 验证
        valid_loss, valid_scores = evaluate(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            split='valid'
        )
        
        # 打印结果
        print('-' * 89)
        print(f'| 轮次 {epoch:3d} | 训练损失 {train_loss:5.4f} | 验证损失 {valid_loss:5.4f} |')
        print(f'| 验证指标 | hits@10: {valid_scores["hits@10"]:.4f} | hits@50: {valid_scores["hits@50"]:.4f} | hits@100: {valid_scores["hits@100"]:.4f} |')
        print(f'| 验证指标 | map@10: {valid_scores["map@10"]:.4f} | map@50: {valid_scores["map@50"]:.4f} | map@100: {valid_scores["map@100"]:.4f} |')
        print('-' * 89)
        
        # 保存每轮的评测指标
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{valid_loss:.6f},"
                    f"{valid_scores['hits@10']:.6f},{valid_scores['hits@50']:.6f},{valid_scores['hits@100']:.6f},"
                    f"{valid_scores['map@10']:.6f},{valid_scores['map@50']:.6f},{valid_scores['map@100']:.6f}\n")
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_metrics = valid_scores.copy()
            torch.save(model.state_dict(), args.save_path)
            print(f"模型已保存到 {args.save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停: {patience} 轮验证损失未改善")
                break
    
    # 打印最佳验证结果
    print('=' * 89)
    print(f'| 最佳验证轮次 {best_epoch:3d} | 验证损失 {best_valid_loss:5.4f} |')
    print(f'| 验证指标 | hits@10: {best_metrics["hits@10"]:.4f} | hits@50: {best_metrics["hits@50"]:.4f} | hits@100: {best_metrics["hits@100"]:.4f} |')
    print(f'| 验证指标 | map@10: {best_metrics["map@10"]:.4f} | map@50: {best_metrics["map@50"]:.4f} | map@100: {best_metrics["map@100"]:.4f} |')
    print('=' * 89)
    
    # 加载最佳模型进行测试
    print("加载最佳模型进行测试...")
    model.load_state_dict(torch.load(args.save_path))
    
    # 测试
    test_loss, test_scores = evaluate(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        split='test'
    )
    
    # 打印测试结果
    print('=' * 89)
    print(f'| 测试损失 {test_loss:5.4f} |')
    print(f'| 测试指标 | hits@10: {test_scores["hits@10"]:.4f} | hits@50: {test_scores["hits@50"]:.4f} | hits@100: {test_scores["hits@100"]:.4f} |')
    print(f'| 测试指标 | map@10: {test_scores["map@10"]:.4f} | map@50: {test_scores["map@50"]:.4f} | map@100: {test_scores["map@100"]:.4f} |')
    print('=' * 89)
    
    # 保存测试指标到文件
    test_metrics_file = os.path.join(result_dir, f'test_metrics_{args.data_name}.txt')
    with open(test_metrics_file, 'w') as f:
        f.write("test_loss,hits@10,hits@50,hits@100,map@10,map@50,map@100\n")
        f.write(f"{test_loss:.6f},"
                f"{test_scores['hits@10']:.6f},{test_scores['hits@50']:.6f},{test_scores['hits@100']:.6f},"
                f"{test_scores['map@10']:.6f},{test_scores['map@50']:.6f},{test_scores['map@100']:.6f}\n")
    
    print(f"测试指标已保存到 {test_metrics_file}")
    
    # 生成预测结果
    print("生成测试集预测结果...")
    predictions, ground_truth, attention_weights = predict(model, data_loader, split='test')
    
    # 保存预测结果
    predictions_file = os.path.join(result_dir, f'predictions_{args.data_name}.txt')
    save_predictions(
        predictions,
        ground_truth,
        data_loader._idx2u,
        file_path=predictions_file
    )
    print(f"预测结果已保存到 {predictions_file}")
    
    print("模型训练和评估完成!")

if __name__ == "__main__":
    main() 