'''A wrapper class for optimizer '''
import numpy as np
import math

class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        
        # 保存初始学习率
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        # 添加最小学习率限制
        self.min_lr = self.initial_lr * 0.01  # 最小学习率为初始学习率的1%

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''

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
        self.last_lr = new_lr
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        return new_lr
