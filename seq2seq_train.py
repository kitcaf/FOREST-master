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
    """计算性能指标，使用组合损失"""
    # 调整预测和目标形状
    pred_flat = pred.view(-1, pred.size(-1))  # [batch_size*tgt_len, user_size]
    gold_flat = gold.contiguous().view(-1)    # [batch_size*tgt_len]
    
    # 计算交叉熵损失
    ce_loss = crit(pred_flat, gold_flat)
    
    # 计算集合损失
    set_loss_val = set_loss(pred, gold)
    
    # 组合损失
    combined_loss = ce_loss + 0.5 * set_loss_val
    
    return combined_loss, {}  # 不再返回准确率

def calculate_metrics(pred, gold, k_list=[10, 50, 100]):
    """计算MAP@k和Hits@k指标"""
    batch_size = gold.size(0)
    scores = {f'hits@{k}': 0.0 for k in k_list}
    scores.update({f'map@{k}': 0.0 for k in k_list})
    
    for i in range(batch_size):
        # 找到目标序列的有效部分（跳过PAD和EOS）
        valid_gold = gold[i][gold[i] != Constants.PAD]
        valid_gold = valid_gold[valid_gold != Constants.EOS]
        
        # 获取预测序列
        pred_probs = pred[i]  # [tgt_len, user_size]
        
        # 对每个位置的用户进行排序
        _, pred_indices = torch.sort(pred_probs, dim=1, descending=True)
        
        # 转换为集合
        gold_set = set(valid_gold.cpu().numpy())
        
        # 计算每个k值的指标
        for k in k_list:
            # 获取前k个预测
            top_k_preds = []
            for t in range(pred_indices.size(0)):
                if len(top_k_preds) >= k:
                    break
                for idx in pred_indices[t][:k]:
                    idx_val = idx.item()
                    if idx_val not in top_k_preds and idx_val != Constants.PAD and idx_val != Constants.EOS:
                        top_k_preds.append(idx_val)
                        if len(top_k_preds) >= k:
                            break
            
            # 计算Hits@k
            hits = 0
            for p in top_k_preds:
                if p in gold_set:
                    hits = 1
                    break
            scores[f'hits@{k}'] += hits
            
            # 计算MAP@k
            ap = 0.0
            hits = 0.0
            for i, p in enumerate(top_k_preds):
                if p in gold_set:
                    hits += 1
                    ap += hits / (i + 1)
            
            if len(gold_set) > 0:
                ap = ap / min(len(gold_set), k)
                scores[f'map@{k}'] += ap
    
    # 计算平均值
    for k in k_list:
        scores[f'hits@{k}'] /= batch_size
        scores[f'map@{k}'] /= batch_size
    
    return scores

def train_epoch(model, data_loader, optimizer, crit, device, k_list=[10, 50, 100], gradient_accumulation_steps=4):
    """训练一个epoch，使用梯度累积减少内存使用"""
    model.train()
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            optimizer.update_learning_rate()
            optimizer.zero_grad()
            
            # 强制垃圾回收
            gc.collect()
            torch.cuda.empty_cache()
    
    # 处理最后一批次（如果有）
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.update_learning_rate()
        optimizer.zero_grad()
    
    # 计算平均值
    for k in k_list:
        epoch_metrics[f'hits@{k}'] /= n_batches
        epoch_metrics[f'map@{k}'] /= n_batches
    
    return total_loss / n_batches, epoch_metrics

def evaluate(model, data_loader, crit, device, mode='valid', k_list=[10, 50, 100]):
    """评估模型"""
    model.eval()
    
    total_loss = 0
    metrics_dict = {f'hits@{k}': 0.0 for k in k_list}
    metrics_dict.update({f'map@{k}': 0.0 for k in k_list})
    n_batches = 0
    
    with torch.no_grad():
        batches = data_loader.get_valid_batches() if mode == 'valid' else data_loader.get_test_batches()
        for batch in tqdm(batches, desc=f"Evaluating on {mode}"):
            # 获取数据
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            time_intervals = batch['time_intervals'].to(device)
            
            # 前向传播
            output = model(src, src_lengths, tgt, time_intervals)
            
            # 计算损失
            loss, _ = get_performance(output, tgt, crit)
            
            # 计算评价指标
            batch_metrics = calculate_metrics(output, tgt, k_list)
            
            # 累加损失和指标
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_dict[k] += v
            n_batches += 1
            
            # 清除不需要的变量以节省内存
            del src, tgt, src_lengths, time_intervals, output
            torch.cuda.empty_cache()
    
    # 计算平均值
    avg_loss = total_loss / n_batches
    for k in metrics_dict:
        metrics_dict[k] /= n_batches
    
    return avg_loss, metrics_dict

def test(model, data_loader, device, k_list=[10, 50, 100], input_ratio=0.5, max_output_len=100):
    """测试模型并计算评价指标"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader.get_test_batches(), desc="Testing"):
            # 获取数据
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            time_intervals = batch['time_intervals'].to(device)
            
            # 生成序列
            generated_seq, generated_probs = model.generate(src, src_lengths, max_len=max_output_len, time_intervals=time_intervals)
            
            # 收集预测和目标
            for i in range(generated_seq.size(0)):
                # 找到目标序列的结束位置（跳过BOS和EOS）
                tgt_end = (tgt[i] == Constants.EOS).nonzero(as_tuple=True)[0]
                tgt_end = tgt_end[0].item() if len(tgt_end) > 0 else tgt[i].size(0)
                
                # 收集目标（跳过BOS）
                target = tgt[i, 1:tgt_end].cpu().numpy()
                
                # 找到生成序列的结束位置
                gen_end = (generated_seq[i] == Constants.EOS).nonzero(as_tuple=True)[0]
                gen_end = gen_end[0].item() if len(gen_end) > 0 else generated_seq[i].size(0)
                
                # 收集预测
                pred = generated_seq[i, :gen_end].cpu().numpy()
                
                # 将预测和目标转换为集合，忽略顺序
                target_set = set(target)
                pred_list = []
                # 保留预测中不重复的用户
                for p in pred:
                    if p not in pred_list and p != Constants.PAD and p != Constants.EOS:
                        pred_list.append(p)
                
                all_targets.append(target_set)
                all_preds.append(pred_list)
            
            # 清除不需要的变量以节省内存
            del src, tgt, src_lengths, time_intervals, generated_seq, generated_probs
            torch.cuda.empty_cache()
    
    # 计算评价指标
    scores = {}
    for k in k_list:
        hits_sum = 0
        map_sum = 0
        
        for target_set, pred_list in zip(all_targets, all_preds):
            # 计算hits@k
            pred_k = pred_list[:k]
            hits = 0
            for p in pred_k:
                if p in target_set:
                    hits = 1
                    break
            hits_sum += hits
            
            # 计算MAP@k
            ap = 0.0
            hits = 0.0
            for i, p in enumerate(pred_k):
                if p in target_set:
                    hits += 1
                    ap += hits / (i + 1)
            
            if len(target_set) > 0:
                ap = ap / min(len(target_set), k)
                map_sum += ap
        
        scores[f'hits@{k}'] = hits_sum / len(all_targets)
        scores[f'map@{k}'] = map_sum / len(all_targets)
    
    return scores

def save_results_to_file(results, filename):
    """将结果保存到文件"""
    with open(filename, 'a') as f:
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--data_name', type=str, default='twitter', help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')  # 减小批次大小
    parser.add_argument('--split_ratio', type=float, default=0.5, help='输入序列与目标序列的分割比例')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=32, help='嵌入维度')  # 减小嵌入维度
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')  # 减小隐藏层维度
    parser.add_argument('--n_layers', type=int, default=1, help='Transformer层数')  # 减少层数
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')  # 减少头数
    parser.add_argument('--pf_dim', type=int, default=256, help='前馈网络维度')  # 减小前馈网络维度
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比例')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='教师强制比例')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=32, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--n_warmup_steps', type=int, default=4000, help='预热步数')
    parser.add_argument('--k_list', type=int, nargs='+', default=[10, 50, 100], help='评价指标的k值列表')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    
    # 其他参数
    parser.add_argument('--no_cuda', action='store_true', help='不使用CUDA')
    parser.add_argument('--use_network', action='store_true', help='是否使用社交网络')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--results_file', type=str, default='results.txt', help='结果保存文件')
    parser.add_argument('--input_ratio', type=float, default=0.6, help='测试时输入序列占原始序列的比例')
    parser.add_argument('--max_output_len', type=int, default=100, help='测试时最大输出序列长度')
    parser.add_argument('--max_seq_length', type=int, default=500, help='最大序列长度')  # 减小最大序列长度
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备 - 默认使用CUDA，除非明确指定不使用
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    print(f"Using device: {device}")
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 加载数据
    print("加载数据...")
    data_loader = Seq2SeqDataLoader(
        data_name=args.data_name,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        cuda=False,  # 不在数据加载器中使用CUDA
        shuffle=True,
        loadNE=args.use_network
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
        adj=data_loader.adj_tensor.to(device) if args.use_network and data_loader.adj_tensor is not None else None,
        net_dict=data_loader.adj_dict if args.use_network else None,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        max_seq_length=args.max_seq_length
    ).to(device)
    
    # 如果有预训练嵌入，加载它们
    if args.use_network and hasattr(data_loader, 'embeds') and data_loader.embeds is not None:
        model.encoder.user_embedding.weight.data.copy_(torch.FloatTensor(data_loader.embeds).to(device))
        model.decoder.user_embedding.weight.data.copy_(torch.FloatTensor(data_loader.embeds).to(device))
    
    # 定义损失函数
    weight = torch.ones(data_loader.user_size).to(device)
    weight[Constants.PAD] = 0
    crit = nn.CrossEntropyLoss(weight=weight, reduction='sum')
    
    # 定义优化器
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.hidden_dim, args.n_warmup_steps
    )
    
    # 训练模型
    print("开始训练...")
    best_valid_loss = float('inf')
    best_valid_map = 0.0
    print(f"=== 训练配置 ===")
    print(f"数据集: {args.data_name}")
    print(f"批次大小: {args.batch_size}")
    print(f"嵌入维度: {args.embed_dim}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"Transformer层数: {args.n_layers}")
    print(f"注意力头数: {args.n_heads}")
    print(f"Dropout比例: {args.dropout}")
    print(f"使用社交网络: {args.use_network}")
    print(f"输入序列比例: {args.split_ratio}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"设备: {device}")

    # 记录训练配置
    with open(args.results_file, 'a') as f:
        f.write(f"=== 训练配置 ===\n")
        f.write(f"数据集: {args.data_name}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"嵌入维度: {args.embed_dim}\n")
        f.write(f"隐藏层维度: {args.hidden_dim}\n")
        f.write(f"Transformer层数: {args.n_layers}\n")
        f.write(f"注意力头数: {args.n_heads}\n")
        f.write(f"Dropout比例: {args.dropout}\n")
        f.write(f"使用社交网络: {args.use_network}\n")
        f.write(f"梯度累积步数: {args.gradient_accumulation_steps}\n")
        f.write(f"设备: {device}\n\n")
    
    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1}/{args.n_epochs}")
        
        # 训练
        start_time = time.time()
        train_loss, train_metrics = train_epoch(
            model, data_loader, optimizer, crit, device, 
            args.k_list, args.gradient_accumulation_steps
        )
        train_time = time.time() - start_time
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 验证
        start_time = time.time()
        valid_loss, valid_metrics = evaluate(model, data_loader, crit, device, mode='valid', k_list=args.k_list)
        valid_time = time.time() - start_time
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.2f}s")
        for k in args.k_list:
            print(f"Train Hits@{k}: {train_metrics[f'hits@{k}']:.4f} | Train MAP@{k}: {train_metrics[f'map@{k}']:.4f}")
        
        print(f"Valid Loss: {valid_loss:.4f} | Time: {valid_time:.2f}s")
        for k in args.k_list:
            print(f"Valid Hits@{k}: {valid_metrics[f'hits@{k}']:.4f} | Valid MAP@{k}: {valid_metrics[f'map@{k}']:.4f}")
        
        # 保存结果到文件
        with open(args.results_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.n_epochs}\n")
            f.write(f"Train Loss: {train_loss:.4f} | Time: {train_time:.2f}s\n")
            for k in args.k_list:
                f.write(f"Train Hits@{k}: {train_metrics[f'hits@{k}']:.4f} | Train MAP@{k}: {train_metrics[f'map@{k}']:.4f}\n")
            
            f.write(f"Valid Loss: {valid_loss:.4f} | Time: {valid_time:.2f}s\n")
            for k in args.k_list:
                f.write(f"Valid Hits@{k}: {valid_metrics[f'hits@{k}']:.4f} | Valid MAP@{k}: {valid_metrics[f'map@{k}']:.4f}\n")
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