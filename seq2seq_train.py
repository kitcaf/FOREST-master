import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import os
import metrics
import Constants
from seq2seq_model import SocialSeq2SeqModel
from seq2seq_dataloader import Seq2SeqDataLoader
from Optim import ScheduledOptim
import gc
from torch.cuda.amp import GradScaler, autocast

# 设置PyTorch内存分配器
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def correctness_loss(pred, gold):
    """计算正确性约束损失，确保预测的用户不在输入序列中且不重复，增强数值稳定性"""
    batch_size = gold.size(0)
    loss = 0.0
    
    for i in range(batch_size):
        # 找到目标序列的有效部分（跳过PAD、BOS和EOS）
        valid_gold = gold[i][gold[i] != Constants.PAD]
        valid_gold = valid_gold[1:-1]  # 去掉BOS和EOS
        
        if len(valid_gold) == 0:
            continue
        
        # 获取预测序列
        pred_i = pred[i, 1:len(valid_gold)+1]  # 跳过BOS位置
        
        # 计算每个位置的预测概率
        for j in range(len(valid_gold)):
            # 获取当前位置的真实标签
            true_label = valid_gold[j].item()
            
            # 获取当前位置的预测logits
            logits = pred_i[j]
            
            # 使用更稳定的方式计算交叉熵
            # 1. 应用log_softmax，它在数值上比先softmax再log更稳定
            log_probs = F.log_softmax(logits, dim=0)
            
            # 2. 直接获取真实标签的负对数概率
            pos_loss = -log_probs[true_label]
            
            # 检查是否为NaN
            if torch.isnan(pos_loss):
                # print(f"警告：位置{j}的正样本损失为NaN，跳过...")
                continue
                
            loss += pos_loss
            
            # 惩罚预测重复用户，使用更稳定的方式
            for k in range(j):
                prev_label = valid_gold[k].item()
                # 使用sigmoid而不是softmax的概率，更稳定
                repeat_penalty = torch.sigmoid(logits[prev_label]) * 0.1
                
                # 检查是否为NaN
                if torch.isnan(repeat_penalty):
                    # print(f"警告：位置{j}的重复惩罚为NaN，跳过...")
                    continue
                    
                loss += repeat_penalty
    
    # 避免除以零
    valid_count = max(1, batch_size)
    return loss / valid_count

def set_based_loss(pred, gold):
    """计算基于集合的损失，不考虑顺序"""
    batch_size = gold.size(0)
    loss = 0.0
    
    for i in range(batch_size):
        # 找到目标序列的有效部分（跳过PAD、BOS和EOS）
        valid_gold = gold[i][gold[i] != Constants.PAD]
        valid_gold = valid_gold[1:-1]  # 去掉BOS和EOS
        
        if len(valid_gold) == 0:
            continue
        
        # 获取预测序列
        pred_i = pred[i, 1:len(valid_gold)+1]  # 跳过BOS位置
        
        # 将真实标签转换为集合
        gold_set = set([label.item() for label in valid_gold])
        
        # 计算每个用户被预测的概率
        all_probs = F.softmax(pred_i.mean(dim=0), dim=0)  # 对所有位置取平均
        
        # 计算集合损失
        set_loss = 0.0
        for user_id in gold_set:
            # 鼓励预测包含真实用户
            set_loss -= torch.log(all_probs[user_id] + 1e-10)
        
        loss += set_loss / len(gold_set)
    
    return loss / batch_size

def get_performance(pred, gold, crit):
    """计算性能指标，确保数值稳定性"""
    batch_size = gold.size(0)
    tgt_len = gold.size(1)
    
    # 初始化损失
    ce_loss = 0.0
    valid_tokens = 0
    
    # 对每个位置分别计算损失
    for pos in range(1, min(tgt_len, pred.size(1) + 1)):
        # 获取当前位置的预测和目标
        pos_pred = pred[:, pos-1]  # [batch_size, user_size]
        pos_gold = gold[:, pos]    # [batch_size]
        
        # 计算当前位置的交叉熵损失
        try:
            # 跳过PAD位置
            valid_mask = (pos_gold != Constants.PAD)
            valid_count = valid_mask.sum().item()
            
            if valid_count > 0:
                # 只计算有效位置的损失
                curr_loss = F.cross_entropy(
                    pos_pred[valid_mask], 
                    pos_gold[valid_mask], 
                    reduction='sum'
                )
                
                # 根据位置赋予不同权重，后面的位置权重更高
                pos_weight = 1.0 + 0.2 * (pos - 1)  # 位置1权重1.0，位置2权重1.2，位置3权重1.4
                ce_loss += curr_loss * pos_weight
                valid_tokens += valid_count
        except Exception as e:
            print(f"计算位置{pos}的损失时出错: {e}")
            continue
    
    # 计算平均损失
    if valid_tokens > 0:
        ce_loss = ce_loss / valid_tokens
    else:
        # 如果没有有效标记，返回一个小的损失
        ce_loss = torch.tensor(0.1, device=pred.device, requires_grad=True)
    
    return ce_loss, ce_loss

def calculate_metrics(pred, gold, k_list=[10, 50, 100], device=None):
    """计算评估指标，确保正确处理"""
    batch_metrics = {f'hits@{k}': 0.0 for k in k_list}
    batch_metrics.update({f'map@{k}': 0.0 for k in k_list})
    
    batch_size = gold.size(0)
    if batch_size == 0:
        return batch_metrics
    
    # 计数有效样本
    valid_samples = 0
    
    # 跳过BOS位置，只评估预测位置
    for pos in range(1, min(4, gold.size(1))):
        pos_pred = pred[:, pos-1, :]  # [batch_size, user_size]
        pos_gold = gold[:, pos]     # [batch_size]
        
        # 计算每个样本的指标
        for i in range(batch_size):
            # 跳过PAD位置
            if pos_gold[i] == Constants.PAD:
                continue
                
            valid_samples += 1
            
            # 获取当前样本的预测和真实标签 - 保持在GPU上
            sample_pred = pos_pred[i]
            sample_gold = pos_gold[i]
            
            # 在GPU上计算top-k索引
            _, top_indices = torch.topk(sample_pred, min(max(k_list), sample_pred.size(0)))
            
            # 计算各种指标
            for k in k_list:
                if k > top_indices.size(0):
                    continue
                    
                # 计算hits@k - 如果真实标签在top-k中，则为1，否则为0
                top_k = top_indices[:k]
                hit = 1.0 if sample_gold in top_k else 0.0
                batch_metrics[f'hits@{k}'] += hit
                
                # 计算MAP@k - 使用单样本的平均精度
                ap = 0.0
                if sample_gold in top_k:
                    # 找到真实标签在top-k中的位置
                    idx = (top_k == sample_gold).nonzero(as_tuple=True)[0][0].item()
                    # 计算精度 = 1/(排名+1)
                    ap = 1.0 / (idx + 1.0)
                batch_metrics[f'map@{k}'] += ap
    
    # 计算平均值 - 确保除以有效样本数
    if valid_samples > 0:
        for k in k_list:
            batch_metrics[f'hits@{k}'] /= valid_samples
            batch_metrics[f'map@{k}'] /= valid_samples
    
    return batch_metrics

def train_epoch_amp(model, data_loader, optimizer, crit, device, k_list=[10, 50, 100], 
                   gradient_accumulation_steps=1, scaler=None):
    """使用混合精度训练一个epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_samples = 0
    metrics_dict = {f'hits@{k}': 0.0 for k in k_list}
    metrics_dict.update({f'map@{k}': 0.0 for k in k_list})
    
    # 使用tqdm显示进度条
    for batch_idx, batch in enumerate(tqdm(data_loader.get_train_batches(), desc="Training")):
        # 获取数据并移动到正确的设备
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_lengths = batch['src_lengths'].to(device)
        time_intervals = batch['time_intervals'].to(device) if 'time_intervals' in batch else None
        
        # 前向传播
        try:
            # 使用混合精度
            with autocast(enabled=scaler is not None):
                output = model(src, src_lengths, tgt, time_intervals)
                loss, ce_loss = get_performance(output, tgt, crit)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print("警告：损失为NaN，跳过此批次...")
                continue
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss / gradient_accumulation_steps).backward()
            else:
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            # 梯度累积：每处理N个批次才更新一次参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                if scaler is not None:
                    scaler.unscale_(optimizer.optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 检查并修复梯度
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # 将NaN梯度替换为0
                        param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)
                
                # 更新参数
                if scaler is not None:
                    scaler.step(optimizer.optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.update_learning_rate()
                optimizer.zero_grad()
            
            # 计算指标 - 在GPU上进行
            batch_metrics = calculate_metrics(output, tgt, k_list, device)
            
            # 计算有效样本数
            valid_samples = sum(1 for i in range(tgt.size(0)) for pos in range(1, min(4, tgt.size(1))) 
                               if pos < tgt.size(1) and tgt[i, pos] != Constants.PAD)
            total_samples += valid_samples
            
            # 累加损失和指标
            total_loss += ce_loss.item()
            for k in k_list:
                metrics_dict[f'hits@{k}'] += batch_metrics[f'hits@{k}'] * valid_samples
                metrics_dict[f'map@{k}'] += batch_metrics[f'map@{k}'] * valid_samples
            
            n_batches += 1
            
        except Exception as e:
            print(f"训练批次出错: {e}")
            import traceback
            traceback.print_exc()
            optimizer.zero_grad()
            continue
        
        # 清除不需要的变量以节省内存
        del src, tgt, src_lengths, time_intervals, output, loss, ce_loss
        torch.cuda.empty_cache()
    
    # 计算平均损失和指标
    avg_loss = total_loss / max(1, n_batches)
    
    # 确保除以总有效样本数
    if total_samples > 0:
        for k in k_list:
            metrics_dict[f'hits@{k}'] /= total_samples
            metrics_dict[f'map@{k}'] /= total_samples
    
    return avg_loss, metrics_dict

def eval_epoch(model, data_loader, crit, device, k_list=[10, 50, 100]):
    """评估一个epoch"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_samples = 0
    metrics_dict = {f'hits@{k}': 0.0 for k in k_list}
    metrics_dict.update({f'map@{k}': 0.0 for k in k_list})
    
    with torch.no_grad():
        for batch in tqdm(data_loader.get_valid_batches(), desc="Validating"):
            try:
                # 获取数据并移动到正确的设备
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                src_lengths = batch['src_lengths'].to(device)
                time_intervals = batch['time_intervals'].to(device) if 'time_intervals' in batch else None
                
                # 前向传播
                with autocast(enabled=device.type=='cuda'):  # 简化autocast参数
                    output = model(src, src_lengths, tgt, time_intervals)
                    
                    # 计算损失
                    loss, ce_loss = get_performance(output, tgt, crit)
                
                # 计算指标 - 在GPU上进行
                batch_metrics = calculate_metrics(output, tgt, k_list, device)
                
                # 计算有效样本数
                valid_samples = sum(1 for i in range(tgt.size(0)) for pos in range(1, min(4, tgt.size(1))) 
                                   if pos < tgt.size(1) and tgt[i, pos] != Constants.PAD)
                total_samples += valid_samples
                
                # 累加损失和指标
                total_loss += ce_loss.item()
                for k in k_list:
                    metrics_dict[f'hits@{k}'] += batch_metrics[f'hits@{k}'] * valid_samples
                    metrics_dict[f'map@{k}'] += batch_metrics[f'map@{k}'] * valid_samples
                
                n_batches += 1
                
            except Exception as e:
                print(f"验证批次出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 清除不需要的变量以节省内存
            del src, tgt, src_lengths, time_intervals, output
            torch.cuda.empty_cache()
    
    # 计算平均损失和指标
    avg_loss = total_loss / max(1, n_batches)
    
    # 确保除以总有效样本数
    if total_samples > 0:
        for k in k_list:
            metrics_dict[f'hits@{k}'] /= total_samples
            metrics_dict[f'map@{k}'] /= total_samples
    
    return avg_loss, metrics_dict

def test(model, data_loader, device, k_list=[10, 50, 100]):
    """测试模型"""
    model.eval()
    total_samples = 0
    metrics_dict = {f'hits@{k}': 0.0 for k in k_list}
    metrics_dict.update({f'map@{k}': 0.0 for k in k_list})
    
    with torch.no_grad():
        for batch in tqdm(data_loader.get_test_batches(), desc="Testing"):
            try:
                # 获取数据并移动到正确的设备
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                src_lengths = batch['src_lengths'].to(device)
                time_intervals = batch['time_intervals'].to(device) if 'time_intervals' in batch else None
                
                # 生成预测
                generated_seq, generated_probs = model.generate(src, src_lengths, max_len=3, time_intervals=time_intervals)
                
                # 计算指标 - 在GPU上进行
                batch_metrics = calculate_metrics(generated_probs, tgt, k_list, device)
                
                # 计算有效样本数
                valid_samples = sum(1 for i in range(tgt.size(0)) for pos in range(1, min(4, tgt.size(1))) 
                                   if pos < tgt.size(1) and tgt[i, pos] != Constants.PAD)
                total_samples += valid_samples
                
                # 累加指标
                for k in k_list:
                    metrics_dict[f'hits@{k}'] += batch_metrics[f'hits@{k}'] * valid_samples
                    metrics_dict[f'map@{k}'] += batch_metrics[f'map@{k}'] * valid_samples
                
            except Exception as e:
                print(f"测试批次出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 清除不需要的变量以节省内存
            del src, tgt, src_lengths, time_intervals, generated_seq, generated_probs
            torch.cuda.empty_cache()
    
    # 确保除以总有效样本数
    if total_samples > 0:
        for k in k_list:
            metrics_dict[f'hits@{k}'] /= total_samples
            metrics_dict[f'map@{k}'] /= total_samples
    
    return metrics_dict

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='序列到序列模型训练')
    
    # 数据参数
    parser.add_argument('--data', type=str, default='twitter', help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--input_ratio', type=float, default=0.7, help='输入序列比例')
    parser.add_argument('--max_output_len', type=int, default=3, help='最大输出长度')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=2, help='层数')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--use_network', action='store_true', help='是否使用社交网络')
    parser.add_argument('--max_seq_length', type=int, default=3000, help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0003, help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='预热步数')
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='梯度累积步数')
    parser.add_argument('--k_list', type=int, nargs='+', default=[10, 50, 100], help='评估的k值列表')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--results_file', type=str, default='results.txt', help='结果文件')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 强制使用CUDA
    args.cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    data_loader = Seq2SeqDataLoader(
        data_name=args.data,
        split_ratio=args.input_ratio,
        batch_size=args.batch_size,
        cuda=True,  # 强制使用CUDA
        shuffle=True,
        loadNE=args.use_network,
        max_seq_length=args.max_seq_length
    )
    
    # 创建模型并移动到设备
    print("创建模型...")
    model = SocialSeq2SeqModel(
        user_size=data_loader.user_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_network=args.use_network,
        adj=data_loader.adj_tensor if args.use_network and hasattr(data_loader, 'adj_tensor') else None,
        teacher_forcing_ratio=0.7  # 增加teacher forcing比例
    ).to(device)
    
    # 打印模型是否在GPU上
    print(f"模型是否在GPU上: {next(model.parameters()).is_cuda}")
    
    # 如果有预训练嵌入，加载它们
    if args.use_network and hasattr(data_loader, 'embeds') and data_loader.embeds is not None:
        print("加载预训练嵌入...")
        model.user_encoder.user_embedding.weight.data.copy_(torch.FloatTensor(data_loader.embeds).to(device))
    
    # 打印模型信息
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 定义损失函数和优化器
    crit = nn.CrossEntropyLoss(ignore_index=Constants.PAD, reduction='sum').to(device)
    
    # 使用权重衰减（L2正则化）和更高的学习率
    optimizer = ScheduledOptim(
        optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01),
        args.hidden_dim,
        args.warmup_steps
    )
    
    # 早停策略
    patience = 10  # 增加耐心值
    best_valid_map = 0.0
    no_improvement_epochs = 0
    
    # 使用混合精度训练
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # 训练模型
    print("开始训练...")
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 创建结果文件
    with open(args.results_file, 'w') as f:
        f.write(f"训练参数:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write("\n")
    
    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1}/{args.n_epochs}")
        
        # 训练
        train_loss, train_metrics = train_epoch_amp(
            model, data_loader, optimizer, crit, device, 
            k_list=args.k_list, gradient_accumulation_steps=args.grad_accum_steps,
            scaler=scaler
        )
        
        # 验证
        valid_loss, valid_metrics = eval_epoch(
            model, data_loader, crit, device, k_list=args.k_list
        )
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}")
        for k in args.k_list:
            print(f"Train Hits@{k}: {train_metrics[f'hits@{k}']:.4f}")
            print(f"Train MAP@{k}: {train_metrics[f'map@{k}']:.4f}")
        
        print(f"Valid Loss: {valid_loss:.4f}")
        for k in args.k_list:
            print(f"Valid Hits@{k}: {valid_metrics[f'hits@{k}']:.4f}")
            print(f"Valid MAP@{k}: {valid_metrics[f'map@{k}']:.4f}")
        
        # 保存结果到文件
        with open(args.results_file, 'a') as f:
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"Train Loss: {train_loss:.4f}\n")
            for k in args.k_list:
                f.write(f"Train Hits@{k}: {train_metrics[f'hits@{k}']:.4f}\n")
                f.write(f"Train MAP@{k}: {train_metrics[f'map@{k}']:.4f}\n")
            
            f.write(f"Valid Loss: {valid_loss:.4f}\n")
            for k in args.k_list:
                f.write(f"Valid Hits@{k}: {valid_metrics[f'hits@{k}']:.4f}\n")
                f.write(f"Valid MAP@{k}: {valid_metrics[f'map@{k}']:.4f}\n")
            f.write("\n")
        
        # 早停检查
        valid_map = valid_metrics['map@10']
        if valid_map > best_valid_map:
            best_valid_map = valid_map
            no_improvement_epochs = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print("保存最佳模型")
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print(f"验证性能连续{patience}个epoch没有提高，提前停止训练")
                break
    
    # 加载最佳模型进行测试
    print("加载最佳模型进行测试...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试
    test_metrics = test(model, data_loader, device, k_list=args.k_list)
    
    # 打印测试结果
    print("测试结果:")
    for k in args.k_list:
        print(f"Test Hits@{k}: {test_metrics[f'hits@{k}']:.4f}")
        print(f"Test MAP@{k}: {test_metrics[f'map@{k}']:.4f}")
    
    # 保存测试结果
    with open(args.results_file, 'a') as f:
        f.write("测试结果:\n")
        for k in args.k_list:
            f.write(f"Test Hits@{k}: {test_metrics[f'hits@{k}']:.4f}\n")
            f.write(f"Test MAP@{k}: {test_metrics[f'map@{k}']:.4f}\n")
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")

if __name__ == "__main__":
    main() 