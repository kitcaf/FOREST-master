"""
序列到序列数据加载器
处理社交网络信息扩散预测任务的数据
"""

import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle
import scipy.sparse as sp
import gc
import os
import time

def normalize(mx):
    """行归一化稀疏矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class Seq2SeqDataLoader:
    """序列到序列数据加载器，用于社交网络信息扩散预测"""
    
    def __init__(self, data_name, split_ratio=0.8, batch_size=32, cuda=True, max_seq_length=20):
        """
        初始化数据加载器
        
        参数:
            data_name: 数据集名称/路径
            split_ratio: 训练集比例
            batch_size: 批次大小
            cuda: 是否使用GPU
            max_seq_length: 最大序列长度
        """
        self.data_name = data_name
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.cuda = cuda
        self.max_seq_length = max_seq_length
        
        # 特殊标记
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        
        # 文件路径
        self.train_data_path = f'data/{data_name}/cascadetrain.txt'
        self.valid_data_path = f'data/{data_name}/cascadevalid.txt'
        self.test_data_path = f'data/{data_name}/cascadetest.txt'
        self.u2idx_dict_path = f'data/{data_name}/u2idx.pickle'
        self.idx2u_dict_path = f'data/{data_name}/idx2u.pickle'
        self.net_data_path = f'data/{data_name}/edges.txt'
        self.embed_dim = 128  # 默认嵌入维度
        self.embed_file_path = f'data/{data_name}/dw{self.embed_dim}.txt'
        
        # 加载数据
        self._load_data()
        
        # 创建数据加载器
        self._create_dataloaders()
        
    def _load_data(self):
        """加载并预处理数据"""
        print(f"加载数据集: {self.data_name}")
        
        # 检查文件是否存在
        self._check_files()
        
        # 加载用户索引
        self._load_user_indices()
        
        # 加载序列数据
        self._load_sequences()
        
        # 加载网络数据
        self._load_network_data()
    
    def _check_files(self):
        """检查必要的文件是否存在"""
        required_files = [
            self.train_data_path,
            self.valid_data_path, 
            self.test_data_path
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到必要的文件: {file_path}")
        
        print("所有必要的数据文件都已找到")
    
    def _load_user_indices(self):
        """加载用户索引映射"""
        try:
            # 尝试加载现有的用户索引
            if os.path.exists(self.u2idx_dict_path) and os.path.exists(self.idx2u_dict_path):
                with open(self.u2idx_dict_path, 'rb') as f:
                    self.user_to_idx = pickle.load(f)
                with open(self.idx2u_dict_path, 'rb') as f:
                    self.idx_to_user = pickle.load(f)
                print(f"从文件加载用户索引映射，共 {len(self.user_to_idx)} 个用户")
                # 设置用户大小
                self.user_size = len(self.user_to_idx)
            else:
                # 创建新的用户索引
                self.user_to_idx = {'<PAD>': self.PAD_token, '<unk>': self.UNK_token, '<s>': self.SOS_token, '</s>': self.EOS_token}
                self._build_user_index()
        except Exception as e:
            print(f"加载用户索引时出错: {e}")
            # 确保用户大小被定义，即使出错
            self.user_size = 4  # 最小值，只包含特殊标记
    
    def _build_user_index(self):
        """构建用户索引映射"""
        self.user_to_idx = {}
        self.idx_to_user = []
        
        # 收集所有用户
        user_set = set()
        
        # 从训练集收集
        for line in open(self.train_data_path):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, _ = chunk.split(',')
                user_set.add(user)
        
        # 从验证集收集
        for line in open(self.valid_data_path):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, _ = chunk.split(',')
                user_set.add(user)
        
        # 从测试集收集
        for line in open(self.test_data_path):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, _ = chunk.split(',')
                user_set.add(user)
        
        # 构建索引
        pos = 0
        self.user_to_idx['<blank>'] = pos
        self.idx_to_user.append('<blank>')
        pos += 1
        self.user_to_idx['</s>'] = pos
        self.idx_to_user.append('</s>')
        pos += 1
        self.user_to_idx['<s>'] = pos
        self.idx_to_user.append('<s>')
        pos += 1
        self.user_to_idx['<unk>'] = pos
        self.idx_to_user.append('<unk>')
        pos += 1
        
        for user in user_set:
            self.user_to_idx[user] = pos
            self.idx_to_user.append(user)
            pos += 1
        
        self.user_size = len(self.user_to_idx)
        print(f"用户词典大小: {self.user_size}")
        
        # 保存索引
        with open(self.u2idx_dict_path, 'wb') as handle:
            pickle.dump(self.user_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.idx2u_dict_path, 'wb') as handle:
            pickle.dump(self.idx_to_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_sequences(self):
        """加载序列数据"""
        print("加载序列数据...")
        
        # 加载训练、验证和测试数据
        train_cascades, train_timestamps = self._read_cascades(self.train_data_path)
        valid_cascades, valid_timestamps = self._read_cascades(self.valid_data_path)
        test_cascades, test_timestamps = self._read_cascades(self.test_data_path)
        
        # 存储数据
        self.train_cascades = train_cascades
        self.valid_cascades = valid_cascades
        self.test_cascades = test_cascades
        
        # 计算时间间隔
        self.train_intervals = self._calculate_time_intervals(train_timestamps)
        self.valid_intervals = self._calculate_time_intervals(valid_timestamps)
        self.test_intervals = self._calculate_time_intervals(test_timestamps)
        
        print(f"训练集大小: {len(self.train_cascades)}")
        print(f"验证集大小: {len(self.valid_cascades)}")
        print(f"测试集大小: {len(self.test_cascades)}")
    
    def _calculate_time_intervals(self, timestamps):
        """计算时间间隔"""
        intervals = []
        
        for ts in timestamps:
            if not ts:  # 如果时间戳列表为空
                intervals.append([])
                continue
                
            interval = [0.0]  # 第一个用户的时间间隔为0
            
            for i in range(1, len(ts)):
                # 计算相对于上一个时间的间隔
                interval.append(ts[i] - ts[i-1])
            
            intervals.append(interval)
        
        return intervals
    
    def _read_cascades(self, file_path, max_len=None):
        """读取级联数据"""
        cascades = []
        timestamps = []
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，返回空列表")
            return [], []
        
        for line in open(file_path):
            if len(line.strip()) == 0:
                continue
                
            chunks = line.strip().split()
            cascade = []
            timestamp = []
            
            for chunk in chunks:
                if ',' in chunk:  # 确保格式正确
                    try:
                        user, time = chunk.split(',')
                        
                        # 将用户名转换为ID
                        if user in self.user_to_idx:
                            user_id = self.user_to_idx[user]
                        else:
                            user_id = self.user_to_idx.get('<unk>', 0)
                        
                        # 将时间转换为浮点数
                        time = float(time)
                        
                        cascade.append(user_id)
                        timestamp.append(time)
                    except Exception as e:
                        print(f"警告: 解析 '{chunk}' 时出错: {e}")
            
            # 如果级联为空，跳过
            if not cascade:
                continue
                
            # 如果指定了最大长度，截断过长的级联
            if max_len is not None and len(cascade) > max_len:
                cascade = cascade[:max_len]
                timestamp = timestamp[:max_len]
            
            cascades.append(cascade)
            timestamps.append(timestamp)
        
        return cascades, timestamps
    
    def _load_network_data(self):
        """加载社交网络数据，构建邻接矩阵"""
        print("加载社交网络数据...")
        
        # 确保user_size已定义
        if not hasattr(self, 'user_size') or self.user_size is None:
            print("警告: 用户大小未定义，使用用户字典长度")
            self.user_size = len(self.user_to_idx) if hasattr(self, 'user_to_idx') else 4
        
        # 检查网络数据文件是否存在
        if not os.path.exists(self.net_data_path):
            print(f"警告: 找不到社交网络数据文件 {self.net_data_path}")
            self.adj_tensor = None
            self.adj_dict = {}
            self.embeds = None
            return
        
        # 创建邻接矩阵
        adj = sp.lil_matrix((self.user_size, self.user_size))
        self.adj_dict = {}
        
        # 读取边数据
        edge_count = 0
        try:
            with open(self.net_data_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 2:
                        continue
                        
                    user1, user2 = parts
                    
                    # 检查用户是否在词典中
                    if user1 in self.user_to_idx and user2 in self.user_to_idx:
                        idx1 = self.user_to_idx[user1]
                        idx2 = self.user_to_idx[user2]
                        
                        # 添加边（无向图）
                        adj[idx1, idx2] = 1
                        adj[idx2, idx1] = 1
                        edge_count += 1
                        
                        # 记录邻居关系
                        if idx1 not in self.adj_dict:
                            self.adj_dict[idx1] = []
                        if idx2 not in self.adj_dict:
                            self.adj_dict[idx2] = []
                            
                        self.adj_dict[idx1].append(idx2)
                        self.adj_dict[idx2].append(idx1)
            
            # 归一化邻接矩阵
            adj = normalize(adj)
            
            # 转换为PyTorch稀疏张量
            self.adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
            if self.cuda:
                self.adj_tensor = self.adj_tensor.cuda()
            
            print(f"社交网络加载完成，共有 {len(self.adj_dict)} 个有连接的用户，{edge_count} 条边")
            
            # 检查预训练嵌入文件是否存在
            if os.path.exists(self.embed_file_path):
                try:
                    self.embeds = self._load_pretrained_embeds()
                    print(f"预训练嵌入加载完成，形状: {self.embeds.shape}")
                except Exception as e:
                    print(f"加载预训练嵌入失败: {e}")
                    self.embeds = None
            else:
                print(f"警告: 找不到预训练嵌入文件 {self.embed_file_path}")
                self.embeds = None
        except Exception as e:
            print(f"加载社交网络数据时出错: {e}")
            self.adj_tensor = None
            self.adj_dict = {}
            self.embeds = None
    
    def _load_pretrained_embeds(self):
        """加载预训练嵌入"""
        embeds = np.zeros((self.user_size, self.embed_dim))
        
        with open(self.embed_file_path, 'r') as f:
            # 跳过第一行
            f.readline()
            
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue
                    
                user = parts[0]
                if user in self.user_to_idx:
                    idx = self.user_to_idx[user]
                    vector = np.array([float(x) for x in parts[1:]])
                    
                    # 确保维度匹配
                    if len(vector) == self.embed_dim:
                        embeds[idx] = vector
                    else:
                        print(f"警告: 用户 {user} 的嵌入维度 ({len(vector)}) 与预期 ({self.embed_dim}) 不匹配")
        
        return embeds
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        print("创建数据批次...")
        
        # 训练批次
        self.train_batches = []
        train_shuffled_indices = list(range(len(self.train_cascades)))
        random.shuffle(train_shuffled_indices)
        
        for i in range(0, len(train_shuffled_indices), self.batch_size):
            # 确保索引有效
            batch_indices = train_shuffled_indices[i:i+self.batch_size]
            
            # 收集批次数据
            batch_cascades = [self.train_cascades[idx] for idx in batch_indices]
            batch_intervals = [self.train_intervals[idx] for idx in batch_indices]
            
            batch = self._create_batch(batch_cascades, batch_intervals)
            if batch is not None:
                self.train_batches.append(batch)
        
        # 验证批次
        self.valid_batches = []
        for i in range(0, len(self.valid_cascades), self.batch_size):
            # 确保索引有效
            end_idx = min(i + self.batch_size, len(self.valid_cascades))
            
            batch = self._create_batch(
                self.valid_cascades[i:end_idx], 
                self.valid_intervals[i:end_idx]
            )
            if batch is not None:
                self.valid_batches.append(batch)
        
        # 测试批次
        self.test_batches = []
        for i in range(0, len(self.test_cascades), self.batch_size):
            # 确保索引有效
            end_idx = min(i + self.batch_size, len(self.test_cascades))
            
            batch = self._create_batch(
                self.test_cascades[i:end_idx], 
                self.test_intervals[i:end_idx]
            )
            if batch is not None:
                self.test_batches.append(batch)
        
        print(f"创建了 {len(self.train_batches)} 个训练批次")
        print(f"创建了 {len(self.valid_batches)} 个验证批次")
        print(f"创建了 {len(self.test_batches)} 个测试批次")

    def _create_batch(self, cascades, intervals):
        """
        创建单个批次
        
        参数:
            cascades: 级联列表
            intervals: 时间间隔列表
        """
        src_seqs = []
        tgt_seqs = []
        src_intervals = []
        src_lengths = []
        
        for idx, (cascade, interval) in enumerate(zip(cascades, intervals)):
            # 跳过太短的级联
            if len(cascade) < 4:  # 至少需要1个输入和3个输出
                continue
            
            # 确保interval长度与cascade匹配
            if len(interval) < len(cascade):
                # 如果时间间隔不足，用1.0填充
                interval = interval + [1.0] * (len(cascade) - len(interval))
            
            # 动态分割点 - 使用split_ratio作为分割比例，但至少保留3个节点作为目标
            split_point = max(1, min(len(cascade) - 3, int(len(cascade) * self.split_ratio)))
            
            # 创建源序列和目标序列
            src = cascade[:split_point]
            
            # 限制源序列长度，防止过长
            if len(src) > self.max_seq_length:
                src = src[-self.max_seq_length:]  # 只保留最后max_seq_length个节点
                interval = interval[-self.max_seq_length:]  # 相应地调整时间间隔
            
            # 目标序列：BOS + 后续3个节点 + EOS
            tgt_nodes = cascade[split_point:split_point+3]
            # 如果目标节点不足3个，用PAD填充
            while len(tgt_nodes) < 3:
                tgt_nodes.append(Constants.PAD)
            
            tgt = [Constants.BOS] + tgt_nodes + [Constants.EOS]
            
            # 获取源序列的时间间隔
            src_interval = interval[:len(src)]
            
            # 确保时间间隔长度与源序列长度一致
            if len(src_interval) < len(src):
                # 如果时间间隔不足，用1.0填充
                src_interval = src_interval + [1.0] * (len(src) - len(src_interval))
            
            # 对时间间隔进行归一化处理
            max_interval = max(src_interval) if src_interval else 1.0
            # 修复除零错误：确保max_interval不为零
            if max_interval == 0:
                max_interval = 1.0
            normalized_interval = [i/max_interval for i in src_interval]
            
            # 记录源序列长度
            src_lengths.append(len(src))
            
            src_seqs.append(src)
            tgt_seqs.append(tgt)
            src_intervals.append(normalized_interval)
        
        # 如果没有有效的级联，返回空批次
        if not src_seqs:
            return None
        
        # 找到最长的级联
        max_src_len = max(len(s) for s in src_seqs)
        max_tgt_len = 5  # BOS + 3个节点 + EOS
        
        # 填充序列
        for i in range(len(src_seqs)):
            src_seqs[i] = src_seqs[i] + [Constants.PAD] * (max_src_len - len(src_seqs[i]))
            tgt_seqs[i] = tgt_seqs[i] + [Constants.PAD] * (max_tgt_len - len(tgt_seqs[i]))
            src_intervals[i] = src_intervals[i] + [0.0] * (max_src_len - len(src_intervals[i]))
        
        # 转换为张量
        src_tensor = torch.LongTensor(src_seqs)
        tgt_tensor = torch.LongTensor(tgt_seqs)
        src_lengths_tensor = torch.LongTensor(src_lengths)
        src_intervals_tensor = torch.FloatTensor(src_intervals)
        
        if self.cuda:
            src_tensor = src_tensor.cuda()
            tgt_tensor = tgt_tensor.cuda()
            src_lengths_tensor = src_lengths_tensor.cuda()
            src_intervals_tensor = src_intervals_tensor.cuda()
        
        return {
            'src': src_tensor,
            'tgt': tgt_tensor,
            'src_lengths': src_lengths_tensor,
            'time_intervals': src_intervals_tensor
        }
    
    def get_train_batches(self):
        """获取训练批次"""
        return self.train_batches
    
    def get_valid_batches(self):
        """获取验证批次"""
        return self.valid_batches
    
    def get_test_batches(self):
        """获取测试批次"""
        return self.test_batches

    def get_adj_tensor(self):
        """获取邻接矩阵张量"""
        return self.adj_tensor 