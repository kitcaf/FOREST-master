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
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
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
    计算单个样本的Hits@k，添加安全检查
    """
    # 确保输入格式正确
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().item()  # 转换为标量
    
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
    计算单个样本的Average Precision@k，添加安全检查
    """
    # 确保输入格式正确
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().item()  # 转换为标量
    
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
        return apk(actual, top_indices, k)
    except Exception as e:
        print(f"MAP计算出错: {e}")
        return 0.0


def mean_rank(y_prob, y):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y):
        ranks += [n_classes - rankdata(p_, method='max')[y_]]

    return sum(ranks) / float(len(ranks))


def portfolio(pred, gold, k_list=[1,5,10,20]):
    scores_len = 0
    y_prob=[]
    y=[]
    for i in range(gold.shape[0]): # predict counts
        if gold[i]!=Constants.PAD:
            scores_len+=1.0
            y_prob.append(pred[i])
            y.append(gold[i])
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = hits_k(y_prob, y, k=k)
        scores['map@' + str(k)] = mapk(y_prob, y, k=k)

    return scores, scores_len
