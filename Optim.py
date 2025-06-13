"""
优化器和学习率调度器
"""
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def get_optimizer(model_params, optimizer_type='adam', lr=0.001, weight_decay=0):
    """
    获取优化器
    
    参数:
        model_params: 模型参数
        optimizer_type: 优化器类型，支持'adam', 'sgd', 'rmsprop'
        lr: 学习率
        weight_decay: 权重衰减
        
    返回:
        optimizer: PyTorch优化器
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

def get_scheduler(optimizer, scheduler_type='step', step_size=5, gamma=0.5, patience=2, factor=0.5):
    """
    获取学习率调度器
    
    参数:
        optimizer: PyTorch优化器
        scheduler_type: 调度器类型，支持'step', 'plateau'
        step_size: 学习率调整步长（用于StepLR）
        gamma: 学习率衰减因子（用于StepLR）
        patience: 容忍轮数（用于ReduceLROnPlateau）
        factor: 学习率衰减因子（用于ReduceLROnPlateau）
        
    返回:
        scheduler: PyTorch学习率调度器
    """
    if scheduler_type.lower() == 'step':
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
