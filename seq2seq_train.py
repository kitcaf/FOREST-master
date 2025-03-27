import argparse
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import os
import metrics
import Constants
from seq2seq_model import ImprovedSeq2SeqModel
from seq2seq_dataloader import Seq2SeqDataLoader
from Optim import ScheduledOptim
import gc

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

def set_loss(pred, gold):
    """计算集合损失，不考虑顺序"""
    batch_size = gold.size(0)
    tgt_len = gold.size(1)
    loss = 0.0
    
    for i in range(batch_size):
        # 找到目标序列的有效部分（跳过PAD和EOS）
        valid_gold = gold[i][gold[i] != Constants.PAD]
        valid_gold = valid_gold[valid_gold != Constants.EOS]
        
        # 找到预测序列的有效部分
        pred_i = pred[i].max(1)[1]
        valid_pred = pred_i[pred_i != Constants.PAD]
        valid_pred = valid_pred[valid_pred != Constants.EOS]
        
        # 转换为集合
        gold_set = set(valid_gold.cpu().numpy())
        pred_set = set(valid_pred.cpu().numpy())
        
        # 计算Jaccard相似度
        intersection = len(gold_set.intersection(pred_set))
        union = len(gold_set.union(pred_set))
        
        # 损失是1减去Jaccard相似度
        if union > 0:
            loss += 1 - (intersection / union)
    
    return loss / batch_size

def get_performance(pred, gold, crit):
    """计算性能指标，使用组合损失，增加数值稳定性"""
    # 调整预测和目标形状
    pred_flat = pred.view(-1, pred.size(-1))  # [batch_size*tgt_len, user_size]
    gold_flat = gold.contiguous().view(-1)    # [batch_size*tgt_len]
    
    # 检查预测值是否包含NaN
    if torch.isnan(pred_flat).any():
        print("警告：预测值包含NaN，正在应用修复...")
        pred_flat = torch.nan_to_num(pred_flat, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 计算交叉熵损失
    ce_loss = crit(pred_flat, gold_flat)
    
    # 计算集合损失
    set_loss_val = set_loss(pred, gold)
    
    # 组合损失
    combined_loss = ce_loss + set_loss_val
    
    # 检查损失是否为NaN
    if torch.isnan(combined_loss):
        print("警告：损失为NaN，使用仅交叉熵损失...")
        return ce_loss, ce_loss
    
    return combined_loss, ce_loss

def calculate_metrics(pred, gold, k_list=[10, 50, 100]):
    """使用metrics.py中的函数计算MAP@k和Hits@k指标"""
    batch_size = gold.size(0)
    metrics_dict = {f'hits@{k}': 0.0 for k in k_list}
    metrics_dict.update({f'map@{k}': 0.0 for k in k_list})
    
    # 只考虑目标序列中的3个实际节点（跳过BOS和EOS）
    for i in range(batch_size):
        # 找到目标序列中的有效部分
        valid_gold = gold[i][gold[i] != Constants.PAD]
        valid_gold = valid_gold[1:-1]  # 去掉BOS和EOS
        
        if len(valid_gold) == 0:
            continue
        
        # 获取预测序列的概率分布
        pred_probs = pred[i, 1:len(valid_gold)+1]  # 跳过第一个位置（对应BOS的预测）
        
        # 确保预测和目标的形状正确
        if pred_probs.size(0) == 0:
            continue
        
        # 对每个位置单独计算指标
        for pos in range(len(valid_gold)):
            pos_pred = pred_probs[pos]  # 当前位置的预测分布
            pos_gold = valid_gold[pos]  # 当前位置的真实标签
            
            # 使用metrics.py中的函数计算指标
            for k in k_list:
                try:
                    # 计算hits@k
                    hits = metrics.hits_k(pos_pred, pos_gold, k=k)
                    metrics_dict[f'hits@{k}'] += hits / len(valid_gold)
                    
                    # 计算MAP@k
                    mapk = metrics.mapk(pos_pred, pos_gold, k=k)
                    metrics_dict[f'map@{k}'] += mapk / len(valid_gold)
                except Exception as e:
                    print(f"Error in metrics calculation: {e}")
                    print(f"pos_pred shape: {pos_pred.shape}, pos_gold: {pos_gold.item() if isinstance(pos_gold, torch.Tensor) else pos_gold}, k: {k}")
                    continue
    
    # 计算平均值
    for k in k_list:
        metrics_dict[f'hits@{k}'] /= batch_size
        metrics_dict[f'map@{k}'] /= batch_size
    
    return metrics_dict

def train_epoch(model, data_loader, optimizer, crit, device, k_list=[10, 50, 100], gradient_accumulation_steps=4):
    """训练一个epoch，使用梯度累积减少内存使用"""
    model.train()
    
    # 打印设备信息
    print(f"训练设备: {device}")
    
    total_loss = 0
    n_batches = 0
    epoch_metrics = {f'hits@{k}': 0.0 for k in k_list}
    epoch_metrics.update({f'map@{k}': 0.0 for k in k_list})
    
    optimizer.zero_grad()  # 初始化梯度
    
    for batch_idx, batch in enumerate(tqdm(data_loader.get_train_batches(), desc="Training")):
        # 获取数据并移动到正确的设备
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_lengths = batch['src_lengths'].to(device)
        time_intervals = batch['time_intervals'].to(device)
        
        # 检查数据是否在GPU上
        if batch_idx == 0:
            print(f"数据是否在GPU上: {src.is_cuda}")
        
        # 前向传播
        output = model(src, src_lengths, tgt, time_intervals)
        
        # 计算损失
        loss, _ = get_performance(output, tgt, crit)
        loss = loss / gradient_accumulation_steps  # 缩放损失
        
        # 反向传播
        loss.backward()
        
        # 计算指标
        batch_metrics = calculate_metrics(output, tgt, k_list)
        
        # 记录指标
        total_loss += loss.item() * gradient_accumulation_steps  # 恢复原始损失值
        n_batches += 1
        for k in k_list:
            epoch_metrics[f'hits@{k}'] += batch_metrics[f'hits@{k}']
            epoch_metrics[f'map@{k}'] += batch_metrics[f'map@{k}']
        
        # 清除不需要的变量以节省内存
        del src, tgt, src_lengths, time_intervals, output
        torch.cuda.empty_cache()
        
        # 梯度累积：每处理N个批次才更新一次参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            # 更新参数
            optimizer.step()
            optimizer.update_learning_rate()
            optimizer.zero_grad()
            
            # 强制垃圾回收
            gc.collect()
            torch.cuda.empty_cache()
    
    # 处理最后一批次（如果有）
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.update_learning_rate()
        optimizer.zero_grad()
    
    # 计算平均值
    for k in k_list:
        epoch_metrics[f'hits@{k}'] /= n_batches
        epoch_metrics[f'map@{k}'] /= n_batches
    
    return total_loss / n_batches, epoch_metrics

def eval_epoch(model, data_loader, crit, device, k_list=[10, 50, 100]):
    """评估一个epoch"""
    model.eval()
    
    total_loss = 0
    n_batches = 0
    epoch_metrics = {f'hits@{k}': 0.0 for k in k_list}
    epoch_metrics.update({f'map@{k}': 0.0 for k in k_list})
    
    with torch.no_grad():
        for batch in tqdm(data_loader.get_valid_batches(), desc="Validating"):
            # 获取数据并移动到正确的设备
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            time_intervals = batch['time_intervals'].to(device)
            
            # 前向传播
            output = model(src, src_lengths, tgt, time_intervals, teacher_forcing_ratio=0.0)
            
            # 计算损失
            loss, _ = get_performance(output, tgt, crit)
            
            # 计算指标
            batch_metrics = calculate_metrics(output, tgt, k_list)
            
            # 记录指标
            total_loss += loss.item()
            n_batches += 1
            for k in k_list:
                epoch_metrics[f'hits@{k}'] += batch_metrics[f'hits@{k}']
                epoch_metrics[f'map@{k}'] += batch_metrics[f'map@{k}']
            
            # 清除不需要的变量以节省内存
            del src, tgt, src_lengths, time_intervals, output
            torch.cuda.empty_cache()
    
    # 计算平均值
    for k in k_list:
        epoch_metrics[f'hits@{k}'] /= n_batches
        epoch_metrics[f'map@{k}'] /= n_batches
    
    return total_loss / n_batches, epoch_metrics

def test(model, data_loader, device, k_list=[10, 50, 100], input_ratio=0.5, max_output_len=3):
    """测试模型"""
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader.get_test_batches(), desc="Testing"):
            # 获取数据并移动到正确的设备
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            time_intervals = batch['time_intervals'].to(device)
            
            # 生成序列
            generated_seq, generated_probs = model.generate(
                src, src_lengths, max_len=max_output_len, 
                time_intervals=time_intervals
            )
            
            # 收集目标和预测
            for i in range(tgt.size(0)):
                # 找到目标序列中的有效部分（跳过BOS和EOS）
                valid_tgt = tgt[i][tgt[i] != Constants.PAD]
                valid_tgt = valid_tgt[1:-1]  # 去掉BOS和EOS
                
                if len(valid_tgt) == 0:
                    continue
                
                # 收集目标和预测
                all_targets.append(valid_tgt.cpu().numpy())
                all_predictions.append(generated_seq[i].cpu().numpy())
    
    # 计算指标
    scores = {}
    
    # 计算Hits@k和MAP@k
    for k in k_list:
        hits_sum = 0
        map_sum = 0
        
        for pred, target in zip(all_predictions, all_targets):
            # 计算Hits@k
            hits = 0
            for t in target:
                if t in pred[:k]:
                    hits += 1
            hits_sum += hits / len(target)
            
            # 计算MAP@k
            ap = 0
            for i, p in enumerate(pred[:k]):
                if p in target:
                    # 计算精度@i
                    precision = sum([1 for j in range(i+1) if pred[j] in target]) / (i+1)
                    ap += precision
            
            if sum([1 for p in pred[:k] if p in target]) > 0:
                ap /= min(k, len(target))
                map_sum += ap
        
        scores[f'hits@{k}'] = hits_sum / len(all_targets) if all_targets else 0
        scores[f'map@{k}'] = map_sum / len(all_targets) if all_targets else 0
    
    return scores

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--data', type=str, default='twitter', help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--input_ratio', type=float, default=0.5, help='输入序列比例')
    parser.add_argument('--max_output_len', type=int, default=3, help='最大输出长度')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=2, help='层数')
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--pf_dim', type=int, default=256, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--use_network', action='store_true', help='是否使用社交网络')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='教师强制比例')
    parser.add_argument('--max_seq_length', type=int, default=500, help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='预热步数')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--k_list', type=int, nargs='+', default=[10, 50, 100], help='评估的k值列表')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 其他参数
    parser.add_argument('--cuda', action='store_true', default=True, help='是否使用CUDA')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--results_file', type=str, default='results.txt', help='结果保存文件')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查CUDA可用性并设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 打印GPU信息
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA不可用，使用CPU")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    data_loader = Seq2SeqDataLoader(
        data_name=args.data,
        split_ratio=args.input_ratio,
        batch_size=args.batch_size,
        cuda=args.cuda,
        shuffle=True,
        loadNE=args.use_network,
        max_seq_length=args.max_seq_length
    )
    
    # 创建模型并移动到设备
    print("创建模型...")
    model = ImprovedSeq2SeqModel(
        user_size=data_loader.user_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        pf_dim=args.pf_dim,
        dropout=args.dropout,
        use_network=args.use_network,
        adj=data_loader.adj_tensor if args.use_network and hasattr(data_loader, 'adj_tensor') else None,
        net_dict=data_loader.adj_dict if args.use_network and hasattr(data_loader, 'adj_dict') else None,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        max_seq_length=args.max_seq_length
    ).to(device)
    
    # 打印模型是否在GPU上
    print(f"模型是否在GPU上: {next(model.parameters()).is_cuda}")
    
    # 如果有预训练嵌入，加载它们
    if args.use_network and hasattr(data_loader, 'embeds') and data_loader.embeds is not None:
        print("加载预训练嵌入...")
        model.encoder.user_embedding.weight.data.copy_(torch.FloatTensor(data_loader.embeds).to(device))
        model.decoder.user_embedding.weight.data.copy_(torch.FloatTensor(data_loader.embeds).to(device))
    
    # 打印模型信息
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 定义损失函数和优化器
    crit = nn.CrossEntropyLoss(ignore_index=Constants.PAD)
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-8),
        args.hidden_dim,
        args.warmup_steps
    )
    
    # 训练模型
    print("开始训练...")
    best_valid_map = 0.0
    
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
        train_loss, train_metrics = train_epoch(
            model, data_loader, optimizer, crit, device, 
            k_list=args.k_list, gradient_accumulation_steps=args.grad_accum_steps
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
        
        # 保存最佳模型（基于验证集MAP@10）
        valid_map = valid_metrics['map@10']
        if valid_map > best_valid_map:
            best_valid_map = valid_map
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
        
        # 每5个epoch测试一次
        if (epoch + 1) % 5 == 0:
            scores = test(model, data_loader, device, k_list=args.k_list, 
                         input_ratio=args.input_ratio, max_output_len=args.max_output_len)
            print("测试结果:")
            for metric, value in scores.items():
                print(f"{metric}: {value:.4f}")
            
            # 保存测试结果到文件
            with open(args.results_file, 'a') as f:
                f.write(f"Epoch {epoch+1} 测试结果:\n")
                for metric, value in scores.items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
            
            # 清理内存
            gc.collect()
            torch.cuda.empty_cache()
    
    # 加载最佳模型进行最终测试
    print("加载最佳模型进行最终测试...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    scores = test(model, data_loader, device, k_list=args.k_list,
                 input_ratio=args.input_ratio, max_output_len=args.max_output_len)
    print("最终测试结果:")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")
    
    # 保存最终测试结果到文件
    with open(args.results_file, 'a') as f:
        f.write("最终测试结果:\n")
        for metric, value in scores.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")

if __name__ == "__main__":
    main() 