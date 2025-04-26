import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from seq2seq_dataloader import Seq2SeqDataLoader
from seq2seq_model import Seq2SeqModel
from Optim import ScheduledOptim
from metrics import portfolio
import Constants

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 定义参数
def parse_args():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('-data_name', type=str, default='twitter', help='数据集名称')
    parser.add_argument('-split_ratio', type=float, default=0.7, help='输入序列与目标序列的分割比例')
    parser.add_argument('-batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('-max_seq_length', type=int, default=100, help='最大序列长度')
    
    # 模型参数
    parser.add_argument('-embed_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('-hidden_size', type=int, default=128, help='隐藏层大小')
    parser.add_argument('-n_layers', type=int, default=2, help='GRU层数')
    parser.add_argument('-dropout', type=float, default=0.1, help='dropout率')
    
    # 训练参数
    parser.add_argument('-n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('-learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('-clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('-teacher_forcing_ratio', type=float, default=0.5, help='教师强制比例')
    parser.add_argument('-n_warmup_steps', type=int, default=2000, help='预热步数')
    
    # 其他参数
    parser.add_argument('-no_cuda', action='store_true', help='不使用CUDA')
    parser.add_argument('-seed', type=int, default=42, help='随机种子')
    parser.add_argument('-save_path', type=str, default='checkpoints/seq2seq_model.pt', help='模型保存路径')
    parser.add_argument('-log_interval', type=int, default=100, help='日志间隔')
    
    args = parser.parse_args()
    
    # 设置CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    return args

# 训练一个epoch
def train_epoch(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, log_interval, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(data_loader.get_train_batches()):
        # 获取数据
        src = batch['src']
        tgt = batch['tgt']
        src_lengths = batch['src_lengths']
        time_intervals = batch['time_intervals']
        
        # 前向传播
        outputs = model(src, src_lengths, time_intervals, tgt, teacher_forcing_ratio)
        
        # 计算损失
        loss = 0
        for t in range(tgt.size(1) - 1):  # -1 是因为最后一个token是EOS
            loss += criterion(outputs[:, t, :], tgt[:, t+1])
        loss = loss / (tgt.size(1) - 1)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        clip_grad_norm_(model.parameters(), clip)
        
        # 参数更新
        optimizer.step()
        optimizer.update_learning_rate()
        
        total_loss += loss.item()
        
        # 打印日志
        if batch_idx % log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.4f}'.format(
                      epoch, batch_idx, len(data_loader.get_train_batches()),
                      elapsed * 1000 / log_interval,
                      total_loss / log_interval))
            total_loss = 0
            start_time = time.time()
    
    return total_loss / len(data_loader.get_train_batches())

# 评估
def evaluate(model, data_loader, criterion, split='valid', k_list=[10, 50, 100]):
    model.eval()
    total_loss = 0
    total_scores = {f'hits@{k}': 0.0 for k in k_list}
    total_scores.update({f'map@{k}': 0.0 for k in k_list})
    total_len = 0
    
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
            
            # 计算损失
            loss = 0
            for t in range(min(tgt.size(1) - 1, 3)):  # 只考虑前3个预测
                loss += criterion(outputs[:, t, :], tgt[:, t+1])
            loss = loss / min(tgt.size(1) - 1, 3)
            
            total_loss += loss.item()
            
            # 计算评估指标
            for t in range(min(tgt.size(1) - 1, 3)):  # 只考虑前3个预测
                # 注意：outputs和tgt可能是CUDA张量，portfolio函数中会处理
                scores, scores_len = portfolio(outputs[:, t, :], tgt[:, t+1], k_list=k_list)
                for k in k_list:
                    total_scores[f'hits@{k}'] += scores[f'hits@{k}'] * scores_len
                    total_scores[f'map@{k}'] += scores[f'map@{k}'] * scores_len
                total_len += scores_len
    
    # 计算平均值
    avg_loss = total_loss / len(data_batches)
    
    # 计算平均指标
    if total_len > 0:
        for k in k_list:
            total_scores[f'hits@{k}'] /= total_len
            total_scores[f'map@{k}'] /= total_len
    
    return avg_loss, total_scores

# 预测
def predict(model, data_loader, split='test', top_k=3):
    model.eval()
    predictions = []
    ground_truth = []
    
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
            pred = model.predict(src, src_lengths, time_intervals, max_length=3)
            
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
    
    return predictions, ground_truth

# 保存预测结果
def save_predictions(predictions, ground_truth, idx2u, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("预测序列\t真实序列\n")
        for pred, true in zip(predictions, ground_truth):
            # 将ID转换为用户名
            pred_users = [idx2u[p] if p < len(idx2u) else '<unk>' for p in pred]
            true_users = [idx2u[t] if t < len(idx2u) and t != Constants.PAD else '<pad>' for t in true]
            
            # 写入文件
            f.write(f"{' -> '.join(pred_users)}\t{' -> '.join(true_users)}\n")

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存模型的目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 加载数据
    data_loader = Seq2SeqDataLoader(
        data_name=args.data_name,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        cuda=args.cuda,
        shuffle=True,
        loadNE=True,
        max_seq_length=args.max_seq_length
    )
    
    # 创建模型
    has_social_data = hasattr(data_loader, 'adj_tensor') and data_loader.adj_tensor is not None
    has_pretrained_embeds = hasattr(data_loader, 'embeds') and data_loader.embeds is not None
    
    if has_social_data:
        print("使用社交网络关系数据进行模型训练")
    else:
        print("警告: 无社交网络关系数据，模型将只使用扩散序列信息")
        
    if has_pretrained_embeds:
        print("使用预训练的用户嵌入")
    else:
        print("警告: 无预训练用户嵌入，将随机初始化用户嵌入")
    
    model = Seq2SeqModel(
        user_size=data_loader.user_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        adj_tensor=data_loader.adj_tensor if has_social_data else None,
        pretrained_embeds=data_loader.embeds if has_pretrained_embeds else None
    )
    
    if args.cuda:
        model = model.cuda()
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=Constants.PAD)
    
    # 定义优化器
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), lr=args.learning_rate),
        args.embed_dim,
        args.n_warmup_steps
    )
    
    # 打印模型信息
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    best_valid_loss = float('inf')
    patience = 5
    patience_counter = 0
    
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
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"模型已保存到 {args.save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停: {patience} 轮验证损失未改善")
                break
    
    # 加载最佳模型进行测试
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
    
    # 生成预测结果
    predictions, ground_truth = predict(model, data_loader, split='test')
    
    # 保存预测结果
    save_predictions(
        predictions,
        ground_truth,
        data_loader._idx2u,
        file_path=f'results_{args.data_name}.txt'
    )
    print(f"预测结果已保存到 results_{args.data_name}.txt")

if __name__ == "__main__":
    main() 