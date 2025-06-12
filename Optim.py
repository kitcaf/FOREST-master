"""
优化器包装类，提供学习率调度功能
"""

import numpy as np
import math

class ScheduledOptim(object):
    """学习率调度优化器包装类"""

    def __init__(self, optimizer, d_model, n_warmup_steps):
        """
        初始化
        
        参数:
            optimizer: 基础优化器
            d_model: 模型维度，用于计算学习率缩放
            n_warmup_steps: 预热步数
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        
        # 保存初始学习率
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        # 学习率历史，用于监控
        self.lr_history = []
        
        # 添加最小学习率限制
        self.min_lr = self.initial_lr * 0.01  # 最小学习率为初始学习率的1%

    def step(self):
        """执行优化器步骤"""
        self.optimizer.step()

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        """更新学习率，根据当前步数动态调整"""
        self.n_current_steps += 1
        
        # 预热阶段
        if self.n_current_steps <= self.n_warmup_steps:
            # 线性增加学习率
            lr_scale = min(1.0, self.n_current_steps / self.n_warmup_steps)
        else:
            # 改进的余弦退火，更平滑且持续更长时间
            progress = (self.n_current_steps - self.n_warmup_steps) / max(1, 100000 - self.n_warmup_steps)
            # 使用更平滑的余弦退火
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            # 限制最小学习率
            lr_scale = max(self.min_lr / self.initial_lr, lr_scale)
        
        # 计算新学习率
        new_lr = self.initial_lr * lr_scale

        # 保存学习率历史以便绘图
        self.lr_history.append(new_lr)
        self.last_lr = new_lr
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        return new_lr
    
    def get_lr_history(self):
        """获取学习率历史"""
        return self.lr_history
    
    def get_last_lr(self):
        """获取最近的学习率"""
        return self.last_lr if hasattr(self, 'last_lr') else self.initial_lr
