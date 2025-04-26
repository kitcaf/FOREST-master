# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:42:32 2021

@author: Ling Sun
"""

import argparse
import time
import numpy as np 
import Constants
import torch
import torch.nn as nn
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics import Metrics
from HGAT import MSHGAT
from Optim import ScheduledOptim
import random


torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

metric = Metrics()


parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='twitter')
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-initialFeatureSize', type=int, default=64)
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.3)
parser.add_argument('-log', default=None)
parser.add_argument('-save_path', default= "./DiffusionPrediction_twitter.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-pos_emb', type=bool, default=True)

opt = parser.parse_args() 
opt.d_word_vec = opt.d_model
#print(opt)

def get_node(pred):
    '''获取预测的节点，返回最后一行值最大的索引'''
    # 获取最后一行
    last_row = pred[-1]
    # 找到最大值的索引
    max_index = torch.argmax(last_row).item()
    return max_index

# 将长度增加
def append_to_tensor(tensor, value):
    '''将一个数加入到形状为 [1, 长度] 的张量中'''
    # 确保输入的张量是二维的
    assert tensor.dim() == 2 and tensor.size(0) == 1, "输入张量必须是形状为 [1, 长度]"

    # 将数值转换为张量，并调整形状为 [1, 1]
    value_tensor = torch.tensor([[value]], dtype=tensor.dtype, device=tensor.device)

    # 使用 torch.cat 在最后一个维度上拼接
    new_tensor = torch.cat((tensor, value_tensor), dim=1)
    return new_tensor

# 获得torch最后一行最后一个值时间戳并加入到序列时间戳中
def append_timestamp(timestamp_seq):
    '''
    处理时间戳序列并添加新的时间戳
    Args:
        timestamp_seq: 形状为 [1, N] 的时间戳序列张量
    Returns:
        添加了新时间戳的序列张量，形状为 [1, N+1]
    '''
    # 获取最后一个时间戳
    last_timestamp = timestamp_seq[0, -1].item()
    
    # 生成一个随机增量（1到100秒之间）
    time_increment = random.uniform(1, 100)
    
    # 计算新的时间戳
    new_timestamp = last_timestamp + time_increment
    
    # 将新时间戳添加到序列中
    new_seq = append_to_tensor(timestamp_seq, new_timestamp)
    
    return new_seq

def get_performance(crit, pred, gold):

    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def test_epoch(model, validation_data, graph, hypergraph_list, window=1, deep=1, state=0, radio=0.1, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    with torch.no_grad():
        for i, batch in enumerate(validation_data):  #tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            print("Validation batch ", i)
            # prepare data
            tgt, tgt_timestamp, tgt_idx =  batch
            # y_gold = tgt[:, 1:]
            print(f"初始之间的输入序列形状", tgt.shape)
            print(f"初始之间的输入序列时间戳形状", tgt_timestamp.shape)
            
            deep_loop = deep
            window = window
            cas_len = tgt.shape[1] #序列长度
            if (state == 0): # 按照%进行计算
                window = int(round(cas_len * radio))
                if (window <= 1):
                    window = 2
                deep_loop = int(cas_len - window)
            elif (state == 1): # 按照固定长度进行计算
                deep_loop = cas_len - window
            elif (state == 2): 
                window = cas_len - 1
                deep_loop = 1

            cas = tgt.clone()
            print(f"state:{state}, window:{window}, deep_loop:{deep_loop}")
            tgt = cas[:, 0 : window]
            tgt_timestamp = tgt_timestamp[:, 0 : window]
            print(f"输入数据形状{tgt.shape}")
            y_gold = cas[:, 2: window + deep_loop]
            print(f"真实指标形状{y_gold.shape}")

            for i in range(deep_loop):
                # forward
                y_pred = model(tgt, tgt_timestamp, tgt_idx, graph, hypergraph_list)
                if state != 2:
                    tgt_timestamp = append_timestamp(tgt_timestamp)
                    node = get_node(y_pred)
                    # print(f"加入节点{node}")
                    tgt = append_to_tensor(tgt, node)
                    
            print(f"预测数据结果：{y_pred.shape}")
            y_pred = y_pred.detach().cpu().numpy()
            y_gold = y_gold.contiguous().view(-1).detach().cpu().numpy()
            scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words
    save_results(scores, window, deep, state, radio)
    return scores

def save_results(scores, window, deep, state, radio):
    """保存测试结果到文件"""
    try:
        with open('result.txt', 'a') as f:
            f.write("\n" + "="*50 + "\n")  # 添加分隔线
            f.write(f"Test Parameters:\n")
            f.write(f"Window: {window}; Deep: {deep}; State: {state}; Radio: {radio}\n\n")
            
            f.write("Test Results:\n")
            for metric in scores.keys():
                f.write(f"{metric}: {scores[metric]:.4f}\n")  # 保留4位小数
            f.write("="*50 + "\n")  # 添加分隔线
    except IOError as e:
        print(f"写入结果文件时出错: {e}")

def test_model(MSHGAT, data_path):
    
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate, load_dict=True)
    
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)
    
    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size

    model = MSHGAT(opt, dropout = opt.dropout)
    model.load_state_dict(torch.load(opt.save_path), strict=False)
    model.cuda()

    # scores = test_epoch(model, test_data, relation_graph, hypergraph_list, state=0, window=1, deep=1, radio=0.2)
    # scores = test_epoch(model, test_data, relation_graph, hypergraph_list, state=0, window=1, deep=1, radio=0.4)
    # scores = test_epoch(model, test_data, relation_graph, hypergraph_list, state=0, window=1, deep=1, radio=0.6)
    # scores = test_epoch(model, test_data, relation_graph, hypergraph_list, state=0, window=2, deep=1, radio=0.8)
    scores = test_epoch(model, test_data, relation_graph, hypergraph_list, state=1, window=3, deep=1, radio=0.2)

    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))


if __name__ == "__main__": 
    model = MSHGAT  
    # train_model(model, opt.data_name)
    test_model(model, opt.data_name)



