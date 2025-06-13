"""
Graph-Augmented Seq2Seq模型训练和评估主程序
用于社交网络信息扩散预测任务
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import math
import copy
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn.functional as F

from seq2seq_dataloader import Seq2SeqDataLoader
from seq2seq_model import GraphAugmentedSeq2Seq
import metrics
import Constants
from Optim import get_optimizer, get_scheduler

# 创建输出目录
os.makedirs(Constants.OUTPUT_DIR, exist_ok=True)
os.makedirs(Constants.MODELS_DIR, exist_ok=True)

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
    parser.add_argument('-split_ratio', type=float, default=0.7, help='训练集与测试集的分割比例')
    parser.add_argument('-batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('-max_seq_length', type=int, default=100, help='最大序列长度')
    
    # 模型参数
    parser.add_argument('-embed_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('-hidden_size', type=int, default=256, help='隐藏层大小')
    parser.add_argument('-n_layers', type=int, default=2, help='GRU层数')
    parser.add_argument('-dropout', type=float, default=0.2, help='dropout率')
    parser.add_argument('-bidirectional', action='store_true', help='是否使用双向编码器')
    
    # 图神经网络参数
    parser.add_argument('-use_social_graph', action='store_true', help='是否使用社交图特征')
    parser.add_argument('-use_social_correction', action='store_true', help='是否使用社交网络修正')
    parser.add_argument('-disable_all_constraints', action='store_true', help='禁用所有社交约束和图特征，纯序列模型')
    
    # 训练参数
    parser.add_argument('-train', action='store_true', help='是否训练模型')
    parser.add_argument('-evaluate', action='store_true', help='是否评估模型')
    parser.add_argument('-predict', action='store_true', help='是否生成预测')
    parser.add_argument('-n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('-lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('-clip_grad', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('-teacher_forcing_ratio', type=float, default=0.5, help='教师强制比例')
    parser.add_argument('-warmup_steps', type=int, default=4000, help='预热步数')
    parser.add_argument('-weight_decay', type=float, default=1e-5, help='权重衰减（L2正则化）')
    parser.add_argument('-save_model', action='store_true', help='是否保存模型')
    parser.add_argument('-save_history', action='store_true', help='是否保存训练历史')
    parser.add_argument('-load_model', default=None, help='加载预训练模型文件路径')
    
    # 优化器和调度器参数
    parser.add_argument('-optim', type=str, default='adam', choices=['sgd', 'adam', 'adamw'], help='优化器类型')
    parser.add_argument('-scheduler', type=str, default='warmup', choices=['none', 'step', 'warmup'], help='学习率调度器类型')
    parser.add_argument('-decay_factor', type=float, default=0.5, help='学习率衰减因子')
    
    # 调度采样和噪声注入参数
    parser.add_argument('-use_scheduled_sampling', action='store_true', help='是否使用调度采样')
    parser.add_argument('-noise_ratio', type=float, default=Constants.MIN_NOISE_RATIO, help='噪声注入率')
    
    # 预测参数
    parser.add_argument('-use_beam_search', action='store_true', help='是否使用束搜索')
    parser.add_argument('-beam_size', type=int, default=Constants.DEFAULT_BEAM_SIZE, help='束搜索宽度')
    parser.add_argument('-temperature', type=float, default=Constants.DEFAULT_TEMPERATURE, help='温度参数')
    parser.add_argument('-output_file', default=None, help='预测结果输出文件')
    
    # 其他参数
    parser.add_argument('-no_cuda', action='store_true', help='不使用CUDA')
    parser.add_argument('-seed', type=int, default=42, help='随机种子')
    parser.add_argument('-log_interval', type=int, default=10, help='日志间隔')
    
    args = parser.parse_args()
    
    # 设置CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # 如果选择禁用所有约束，则设置相关参数
    if args.disable_all_constraints:
        args.use_social_graph = False
        args.use_social_correction = False
        print("警告: 已禁用所有社交网络约束和图特征，将使用纯序列模型")
    
    # 如果未指定任何操作模式，默认执行训练和评估
    if not (args.train or args.evaluate or args.predict):
        args.train = True
        args.evaluate = True
        args.predict = True
    
    return args

# 训练一个epoch
def train_epoch(model, data_iter, optimizer, criterion, args):
    """
    训练一个epoch
    
    参数:
        model: 模型
        data_iter: 数据迭代器
        optimizer: 优化器
        criterion: 损失函数
        args: 参数
        
    返回:
        epoch_loss: 本轮损失
    """
    model.train()
    total_loss = 0
    
    # 进度条
    pbar = tqdm(enumerate(data_iter), total=len(data_iter), desc="训练中")
    
    for i, batch in pbar:
        # 清除梯度
        optimizer.zero_grad()
        
        # 获取数据
        src_seq = batch['src']
        tgt_seq = batch['tgt']
        
        # 前向传播
        outputs, pred_nodes, last_node = model(src_seq, tgt_seq, 
                                             teacher_forcing_ratio=args.teacher_forcing_ratio,
                                             noise_ratio=args.noise_ratio)
        
        # 计算损失
        loss = 0
        
        # 交叉熵损失
        # 展平输出和目标
        flat_outputs = outputs.view(-1, outputs.size(-1))
        # 跳过BOS标记，从第二个标记开始
        flat_targets = tgt_seq[:, 1:].contiguous().view(-1)
        # 计算交叉熵损失
        ce_loss = criterion(flat_outputs, flat_targets)
        loss += ce_loss
        
        # 社交一致性损失
        if args.use_social_loss and model.has_social_graph():
            # 计算社交一致性损失
            batch_size = src_seq.size(0)
            
            # 对每个预测位置计算社交一致性损失
            social_loss = 0
            for t in range(min(len(pred_nodes), args.max_seq_length - 1)):
                # 获取当前位置的预测节点
                curr_pred = pred_nodes[t]
                # 获取上一个位置的目标节点（作为参考）
                prev_tgt = tgt_seq[:, t]
                # 计算社交一致性损失
                s_loss = model.calculate_social_consistency_loss(curr_pred, prev_tgt)
                if s_loss is not None:
                    social_loss += s_loss
            
            # 如果有社交一致性损失，加到总损失中
            if social_loss > 0:
                loss += args.social_loss_weight * social_loss / len(pred_nodes)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix(loss=loss.item(), avg_loss=total_loss/(i+1))
        
        # 检查是否需要提前停止
        if args.debug and i >= 5:
            print("调试模式：提前停止训练")
            break
    
    return total_loss / len(data_iter)

# 评估
def evaluate(model, data_iter, criterion, args):
    """
    评估模型
    
    参数:
        model: 模型
        data_iter: 数据迭代器
        criterion: 损失函数
        args: 参数
        
    返回:
        avg_loss: 平均损失
        error_rate: 错误率
        metrics: 评估指标
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_iter:
            # 获取数据
            src_seq = batch['src']
            tgt_seq = batch['tgt']
            
            # 前向传播 - 训练模式以获取损失
            outputs, _, _ = model(src_seq, tgt_seq)
            
            # 计算损失
            # 展平输出和目标
            flat_outputs = outputs.view(-1, outputs.size(-1))
            # 跳过BOS标记，从第二个标记开始
            flat_targets = tgt_seq[:, 1:].contiguous().view(-1)
            # 计算交叉熵损失
            loss = criterion(flat_outputs, flat_targets)
            
            # 累计损失
            batch_size = src_seq.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 推理模式获取预测
            pred_seq, _ = model(src_seq, use_beam_search=args.use_beam_search, 
                               beam_size=args.beam_size,
                               use_social_correction=args.use_social_correction)
            
            # 收集预测和目标
            all_preds.append(pred_seq.cpu())
            all_targets.append(tgt_seq.cpu())
    
    # 计算平均损失
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    # 连接所有预测和目标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算错误率 - 第一个预测位置的错误率
    # 跳过BOS标记，比较第一个预测位置
    first_pred = all_preds[:, 1]  # 第一个预测位置
    first_target = all_targets[:, 1]  # 第一个目标位置
    error_rate = (first_pred != first_target).float().mean().item()
    
    # 计算评估指标
    metrics = calculate_metrics(all_preds, all_targets)
    
    return avg_loss, error_rate, metrics

# 预测
def predict(model, data_iter, args):
    """
    使用模型生成预测
    
    参数:
        model: 模型
        data_iter: 测试数据迭代器
        args: 参数
        
    返回:
        predictions: 预测结果
        metrics: 评估指标
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_srcs = []
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定输出文件路径
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"predictions_{timestamp}.txt")
    
    # 确定指标文件路径
    metrics_file = os.path.splitext(output_file)[0] + "_metrics.json"
    
    print(f"开始预测，结果将保存到: {output_file}")
    
    with torch.no_grad():
        for batch in tqdm(data_iter, desc="预测中"):
            # 获取数据
            src_seq = batch['src']
            
            # 如果有目标序列，收集它用于评估
            if 'tgt' in batch:
                tgt_seq = batch['tgt']
                all_targets.append(tgt_seq.cpu())
            
            # 收集源序列
            all_srcs.append(src_seq.cpu())
            
            # 预测
            pred_seq, _ = model(src_seq, use_beam_search=args.use_beam_search, 
                               beam_size=args.beam_size,
                               use_social_correction=args.use_social_correction,
                               temperature=args.temperature)
            
            # 收集预测
            all_preds.append(pred_seq.cpu())
    
    # 连接所有预测、源和目标
    all_preds = torch.cat(all_preds, dim=0)
    all_srcs = torch.cat(all_srcs, dim=0)
    
    # 计算评估指标
    metrics = None
    if all_targets:
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_preds, all_targets)
        
        # 打印评估指标      
        print("测试集评估指标:")
        for k in sorted(metrics.keys()):
            print(f"  {k}: {metrics[k]:.4f}")
        
        # 保存评估指标
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"评估指标已保存到: {metrics_file}")
    
    # 保存预测结果
    with open(output_file, 'w') as f:
        for i in range(all_preds.size(0)):
            src = ' '.join(map(str, all_srcs[i].tolist()))
            pred = ' '.join(map(str, all_preds[i].tolist()))
            f.write(f"Source: {src}\n")
            f.write(f"Prediction: {pred}\n")
            if all_targets:
                tgt = ' '.join(map(str, all_targets[i].tolist()))
                f.write(f"Target: {tgt}\n")
            f.write("\n")
    
    print(f"预测结果已保存到: {output_file}")
    
    return all_preds, metrics

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

def calculate_metrics(pred_seq, tgt_seq, k_values=None):
    """
    计算评估指标
    
    参数:
        pred_seq: [batch_size, seq_len] - 预测序列
        tgt_seq: [batch_size, seq_len] - 目标序列
        k_values: 评估的K值列表，默认为[10, 50, 100]
        
    返回:
        metrics: 包含各种评估指标的字典
    """
    if k_values is None:
        k_values = [10, 50, 100]
    
    batch_size = pred_seq.size(0)
    metrics = {}
    
    # 初始化指标
    for k in k_values:
        metrics[f'hits@{k}'] = 0.0
        metrics[f'map@{k}'] = 0.0
    
    # 计算前三个位置的指标（跳过BOS标记）
    valid_samples = 0
    for pos in range(1, min(4, tgt_seq.size(1))):  # 评估前3个位置 (跳过BOS)
        for i in range(batch_size):
            # 获取当前样本的预测和目标
            if pos < pred_seq.size(1) and pos < tgt_seq.size(1):
                pred = pred_seq[i, pos].item()
                tgt = tgt_seq[i, pos].item()
                
                # 跳过PAD和特殊标记
                if tgt < 4:  # 特殊标记
                    continue
                
                valid_samples += 1
                
                # 假设前k个预测是[pred]（因为我们只有贪婪解码的结果）
                # 对于每个k值计算hits@k和map@k
                for k in k_values:
                    # 如果预测正确，则hits@k=1，否则为0
                    if pred == tgt:
                        metrics[f'hits@{k}'] += 1.0
                        # 如果预测正确，则map@k=1，因为排名为1
                        metrics[f'map@{k}'] += 1.0
    
    # 计算平均指标
    if valid_samples > 0:
        for k in k_values:
            metrics[f'hits@{k}'] /= valid_samples
            metrics[f'map@{k}'] /= valid_samples
    
    return metrics

def train(model, data_loader, optimizer, scheduler, criterion, args):
    """
    训练模型
    
    参数:
        model: 模型
        data_loader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        args: 参数
        
    返回:
        history: 训练历史记录
    """
    print("开始训练...")
    
    # 初始化训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_error_rate': [],
        'val_metrics': []
    }
    
    # 创建保存模型的目录
    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # 最佳验证损失
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(model, data_loader.train_batches, optimizer, criterion, args)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 验证
        val_loss, val_error_rate, val_metrics = evaluate(model, data_loader.valid_batches, criterion, args)
        
        # 更新训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_error_rate'].append(val_error_rate)
        history['val_metrics'].append(val_metrics)
        
        # 打印训练信息
        print('-' * 89)
        print(f'| 轮次 {epoch:3d} | 时间 {time.time() - epoch_start_time:5.2f}s | '
              f'训练损失 {train_loss:.4f} | 验证损失 {val_loss:.4f} | '
              f'错误率 {val_error_rate:.4f}')
        print('-' * 89)
        
        # 打印验证指标
        print("验证指标:")
        for k in sorted(val_metrics.keys()):
            print(f"  {k}: {val_metrics[k]:.4f}")
        
        # 保存最佳模型
        if args.save_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f'model_epoch{epoch}_loss{val_loss:.4f}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")
            
            # 保存训练历史
            history_path = os.path.join(args.save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f)
            print(f"训练历史已保存到 {history_path}")
    
    return history

def create_optimizer(model, lr, weight_decay=0.0):
    """
    创建优化器
    
    参数:
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
        
    返回:
        optimizer: 优化器
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def create_scheduler(optimizer, factor=0.5, patience=2, min_lr=1e-6):
    """
    创建学习率调度器
    
    参数:
        optimizer: 优化器
        factor: 学习率衰减因子
        patience: 容忍的轮数
        min_lr: 最小学习率
        
    返回:
        scheduler: 学习率调度器
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=True
    )

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='序列到序列模型训练与预测')
    
    # 数据参数
    parser.add_argument('--data_name', type=str, default='twitter', help='数据集名称')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_seq_length', type=int, default=20, help='最大序列长度')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--hidden_size', type=int, default=256, help='隐藏层大小')
    parser.add_argument('--n_layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--use_social_graph', action='store_true', help='是否使用社交图')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='教师强制比例')
    parser.add_argument('--noise_ratio', type=float, default=0.1, help='噪声注入率')
    parser.add_argument('--use_social_loss', action='store_true', help='是否使用社交一致性损失')
    parser.add_argument('--social_loss_weight', type=float, default=0.2, help='社交一致性损失权重')
    
    # 预测参数
    parser.add_argument('--use_beam_search', action='store_true', help='是否使用束搜索')
    parser.add_argument('--beam_size', type=int, default=5, help='束搜索大小')
    parser.add_argument('--use_social_correction', action='store_true', help='是否使用社交修正')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    
    # 其他参数
    parser.add_argument('--cuda', action='store_true', help='是否使用GPU')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--save_model', action='store_true', help='是否保存模型')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    parser.add_argument('--load_model', type=str, help='加载模型路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 检查CUDA可用性
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载数据集: {args.data_name}")
    data_loader = Seq2SeqDataLoader(
        data_name=args.data_name,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        cuda=args.cuda,
        max_seq_length=args.max_seq_length
    )
    
    # 打印数据集信息
    print(f"用户数量: {data_loader.user_size}")
    print(f"训练集大小: {len(data_loader.train_cascades)}")
    print(f"验证集大小: {len(data_loader.valid_cascades)}")
    print(f"测试集大小: {len(data_loader.test_cascades)}")
    
    # 检查是否加载了社交图
    has_social_graph = hasattr(data_loader, 'adj_tensor') and data_loader.adj_tensor is not None
    if args.use_social_graph and not has_social_graph:
        print("警告: 请求使用社交图，但未能加载社交图数据。将禁用社交图功能。")
        args.use_social_graph = False
        args.use_social_correction = False
        args.use_social_loss = False
    elif args.use_social_graph and has_social_graph:
        print("成功加载社交图数据。")
    
    # 构建模型
    print("构建模型...")
    model = GraphAugmentedSeq2Seq(
        user_size=data_loader.user_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        adj_tensor=data_loader.adj_tensor if args.use_social_graph else None,
        pretrained_embeds=data_loader.pretrained_embeddings if hasattr(data_loader, 'pretrained_embeddings') else None
    )
    
    # 移动模型到设备
    model = model.to(device)
    
    # 加载预训练模型
    if args.load_model:
        print(f"加载预训练模型: {args.load_model}")
        model.load_state_dict(torch.load(args.load_model, map_location=device))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=Constants.PAD)
    optimizer = create_optimizer(model, args.lr)
    scheduler = create_scheduler(optimizer, factor=0.5, patience=2)
    
    # 训练或测试
    if args.mode == 'train':
        print("开始训练...")
        history = train(model, data_loader, optimizer, scheduler, criterion, args)
        
        # 在测试集上评估
        print("在测试集上评估...")
        test_loss, test_error_rate, test_metrics = evaluate(
            model=model,
            data_iter=data_loader.test_batches,
            criterion=criterion,
            args=args
        )
        
        print(f"测试集损失: {test_loss:.4f}, 错误率: {test_error_rate:.4f}")
        print("测试集指标:")
        for k in sorted(test_metrics.keys()):
            print(f"  {k}: {test_metrics[k]:.4f}")
        
        # 保存测试指标
        metrics_file = os.path.join(args.output_dir, f"{args.data_name}_test_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"测试指标已保存到: {metrics_file}")
        
    else:  # 测试模式
        print("开始测试...")
        predictions, metrics = predict(
            model=model,
            data_iter=data_loader.test_batches,
            args=args
        )
    
    print("完成!")

if __name__ == "__main__":
    main() 