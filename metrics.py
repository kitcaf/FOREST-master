'''
Evaluation metrics functions.
'''
# import math
import numpy as np
import collections
from sklearn.preprocessing import label_binarize
from scipy.stats import rankdata
import Constants
import torch

def _retype(y_prob, y):
    """将输入转换为numpy数组"""
    if not isinstance(y, (collections.Sequence, np.ndarray)):
        y_prob = [y_prob]
        y = [y]
    y_prob = np.array(y_prob)
    y = np.array(y)

    return y_prob, y

def _binarize(y, n_classes=None):
    return label_binarize(y, classes=range(n_classes))

def apk(actual, predicted, k=10):
    """
    计算平均精度@k
    
    参数:
        actual: 实际元素列表
        predicted: 预测元素列表（顺序很重要）
        k: 考虑的预测元素的最大数量
        
    返回:
        score: 平均精度@k
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def hits_k(y_prob, y, k=10):
    """
    计算单个样本的Hits@k
    
    参数:
        y_prob: 预测概率
        y: 真实标签
        k: 考虑的预测元素的最大数量
        
    返回:
        1.0 如果真实标签在前k个预测中，否则0.0
    """
    # 确保输入格式正确
    if isinstance(y_prob, torch.Tensor):
        if y_prob.is_cuda:
            y_prob = y_prob.detach().cpu().numpy()
        else:
            y_prob = y_prob.detach().numpy()
    
    if isinstance(y, torch.Tensor):
        if y.is_cuda:
            y = y.detach().cpu().item()  # 转换为标量
        else:
            y = y.detach().item()
    
    # 确保k不超过预测张量的维度
    effective_k = min(k, len(y_prob))
    if effective_k <= 0:
        return 0.0
    
    # 检查y_prob是否包含NaN或Inf
    if np.isnan(y_prob).any() or np.isinf(y_prob).any():
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1e6, neginf=-1e6)
    
    try:
        # 获取前k个预测的索引
        top_indices = np.argsort(y_prob)[-effective_k:][::-1]
        
        # 如果真实标签在top-k中，则为1，否则为0
        return 1.0 if y in top_indices else 0.0
    except Exception as e:
        print(f"Hits@k计算出错: {e}")
        return 0.0

def mapk(y_prob, y, k=10):
    """
    计算单个样本的Average Precision@k
    
    参数:
        y_prob: 预测概率
        y: 真实标签
        k: 考虑的预测元素的最大数量
        
    返回:
        Average Precision@k
    """
    # 确保输入格式正确
    if isinstance(y_prob, torch.Tensor):
        if y_prob.is_cuda:
            y_prob = y_prob.detach().cpu().numpy()
        else:
            y_prob = y_prob.detach().numpy()
    
    if isinstance(y, torch.Tensor):
        if y.is_cuda:
            y = y.detach().cpu().item()  # 转换为标量
        else:
            y = y.detach().item()
    
    # 确保k不超过预测张量的维度
    effective_k = min(k, len(y_prob))
    if effective_k <= 0:
        return 0.0
    
    # 检查y_prob是否包含NaN或Inf
    if np.isnan(y_prob).any() or np.isinf(y_prob).any():
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1e6, neginf=-1e6)
    
    try:
        # 获取前k个预测的索引
        top_indices = np.argsort(y_prob)[-effective_k:][::-1]
        
        # 计算AP@k
        actual = [y]  # 单个元素的列表
        return apk(actual, top_indices.tolist(), k)
    except Exception as e:
        print(f"MAP计算出错: {e}")
        return 0.0


def mean_rank(y_prob, y):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y):
        ranks += [n_classes - rankdata(p_, method='max')[y_]]

    return sum(ranks) / float(len(ranks))


def portfolio(pred, gold, k_list=[10,50,100]):
    """
    计算多个评估指标
    
    参数:
        pred: 预测输出，可能是CUDA张量 [batch_size, vocab_size]
        gold: 真实标签，可能是CUDA张量 [batch_size]
        k_list: 评估的K值列表
    
    返回:
        scores: 评估分数字典，包含hits@k和map@k
        scores_len: 有效样本数
    """
    scores_len = 0
    y_prob = []
    y = []
    
    # 确保处理的是CPU张量
    if isinstance(pred, torch.Tensor) and pred.is_cuda:
        pred_cpu = pred.detach().cpu()
    else:
        pred_cpu = pred
    
    if isinstance(gold, torch.Tensor) and gold.is_cuda:
        gold_cpu = gold.detach().cpu()
    else:
        gold_cpu = gold
    
    # 收集有效样本
    for i in range(gold_cpu.shape[0]):
        if gold_cpu[i] != Constants.PAD:
            scores_len += 1.0
            if isinstance(pred_cpu, torch.Tensor):
                y_prob.append(pred_cpu[i].numpy())
            else:
                y_prob.append(pred_cpu[i])
            y.append(gold_cpu[i].item() if isinstance(gold_cpu, torch.Tensor) else gold_cpu[i])
    
    scores = {}
    
    # 如果没有有效样本，返回零分
    if scores_len == 0:
        for k in k_list:
            scores['hits@' + str(k)] = 0.0
            scores['map@' + str(k)] = 0.0
        return scores, 0.0
    
    # 计算hits@k和map@k指标
    for k in k_list:
        try:
            hits = []
            maps = []
            for prob, truth in zip(y_prob, y):
                # 计算hits@k
                hit = hits_k(prob, truth, k)
                hits.append(hit)
                
                # 计算map@k
                ap = mapk(prob, truth, k)
                maps.append(ap)
            
            # 计算平均值
            scores['hits@' + str(k)] = sum(hits) / len(hits)
            scores['map@' + str(k)] = sum(maps) / len(maps)
        except Exception as e:
            print(f"计算指标出错 (k={k}): {e}")
            scores['hits@' + str(k)] = 0.0
            scores['map@' + str(k)] = 0.0

    return scores, scores_len
