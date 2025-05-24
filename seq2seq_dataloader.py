import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle
import scipy.sparse as sp
import gc
import os

from utils.graph_utils import normalize, sparse_mx_to_torch_sparse_tensor, construct_hypergraph

class Seq2SeqDataLoader:
    """序列到序列数据加载器"""
    
    def __init__(self, data_name, split_ratio=0.5, batch_size=32, cuda=True, shuffle=True, loadNE=True, max_seq_length=500):
        """
        初始化数据加载器
        
        参数:
            data_name: 数据集名称
            split_ratio: 输入序列与目标序列的分割比例
            batch_size: 批次大小
            cuda: 是否使用CUDA
            shuffle: 是否打乱数据
            loadNE: 是否加载网络嵌入
            max_seq_length: 最大序列长度
        """
        self.data_name = data_name
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.cuda = cuda
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        
        # 打印CUDA设置
        print(f"数据加载器CUDA设置: {self.cuda}")
        
        # 文件路径
        self.train_data_path = f'data/{data_name}/cascade.txt'
        self.valid_data_path = f'data/{data_name}/cascadevalid.txt'
        self.test_data_path = f'data/{data_name}/cascadetest.txt'
        self.u2idx_dict_path = f'data/{data_name}/u2idx.pickle'
        self.idx2u_dict_path = f'data/{data_name}/idx2u.pickle'
        self.net_data_path = f'data/{data_name}/edges.txt'
        self.embed_dim = 32  # 减小嵌入维度
        self.embed_file_path = f'data/{data_name}/dw{self.embed_dim}.txt'
        
        # 加载用户索引
        self._load_user_index()
        
        # 加载级联数据
        self._load_cascades()
        
        # 如果需要，加载网络数据
        if loadNE:
            self._load_network_data()
        
        # 创建数据批次
        self._create_batches()
        
        # 确保时间间隔属性存在
        if not hasattr(self, 'train_intervals'):
            self.train_intervals = [self._create_default_intervals(cascade) for cascade in self.train_cascades]
        
        if not hasattr(self, 'valid_intervals'):
            self.valid_intervals = [self._create_default_intervals(cascade) for cascade in self.valid_cascades]
        
        if not hasattr(self, 'test_intervals'):
            self.test_intervals = [self._create_default_intervals(cascade) for cascade in self.test_cascades]
        
    def _load_user_index(self):
        """加载用户索引映射"""
        try:
            with open(self.u2idx_dict_path, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.idx2u_dict_path, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)
            print(f"用户词典大小: {self.user_size}")
        except:
            print("找不到用户索引文件，正在创建...")
            self._build_user_index()
    
    def _build_user_index(self):
        """构建用户索引映射"""
        self._u2idx = {}
        self._idx2u = []
        
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
        self._u2idx['<blank>'] = pos
        self._idx2u.append('<blank>')
        pos += 1
        self._u2idx['</s>'] = pos
        self._idx2u.append('</s>')
        pos += 1
        self._u2idx['<s>'] = pos
        self._idx2u.append('<s>')
        pos += 1
        self._u2idx['<unk>'] = pos
        self._idx2u.append('<unk>')
        pos += 1
        
        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        
        self.user_size = len(self._u2idx)
        print(f"用户词典大小: {self.user_size}")
        
        # 保存索引
        with open(self.u2idx_dict_path, 'wb') as handle:
            pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.idx2u_dict_path, 'wb') as handle:
            pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_cascades(self):
        """加载级联数据"""
        self.train_cascades, self.train_timestamps = self._read_cascades(self.train_data_path)
        self.valid_cascades, self.valid_timestamps = self._read_cascades(self.valid_data_path)
        self.test_cascades, self.test_timestamps = self._read_cascades(self.test_data_path)
        
        # 计算时间间隔
        self.train_intervals = self._calculate_time_intervals(self.train_timestamps)
        self.valid_intervals = self._calculate_time_intervals(self.valid_timestamps)
        self.test_intervals = self._calculate_time_intervals(self.test_timestamps)
        
        print(f"训练集大小: {len(self.train_cascades)}")
        print(f"验证集大小: {len(self.valid_cascades)}")
        print(f"测试集大小: {len(self.test_cascades)}")
    
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
                        if user in self._u2idx:
                            user_id = self._u2idx[user]
                        else:
                            user_id = self._u2idx.get('<unk>', 0)
                        
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
    
    def _calculate_time_intervals(self, timestamps):
        """计算时间间隔"""
        intervals = []
        
        for ts in timestamps:
            if not ts:  # 如果时间戳列表为空
                intervals.append([])
                continue
                
            interval = [0.0]  # 第一个用户的时间间隔为0
            
            for i in range(1, len(ts)):
                interval.append(ts[i] - ts[i-1])
            
            intervals.append(interval)
        
        return intervals
    
    def _load_network_data(self):
        """加载社交网络数据，并构建DisenIDP风格的超图"""
        print("加载社交网络数据...")
        
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
        with open(self.net_data_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue
                    
                user1, user2 = parts
                
                # 检查用户是否在词典中
                if user1 in self._u2idx and user2 in self._u2idx:
                    idx1 = self._u2idx[user1]
                    idx2 = self._u2idx[user2]
                    
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
        
        # 构建DisenIDP风格的超图
        print("构建DisenIDP风格的超图...")
        # 合并所有级联数据用于构建超图
        all_cascades = self.train_cascades + self.valid_cascades + self.test_cascades
        
        # 使用DisenIDP的超图构建方法
        window_size = 5  # 滑动窗口大小
        self.HG_Item, self.HG_User = construct_hypergraph(all_cascades, self.user_size, window_size)
        
        if self.cuda:
            self.HG_Item = self.HG_Item.cuda()
            self.HG_User = self.HG_User.cuda()
        
        print("超图构建完成")
        
        # 检查预训练嵌入文件是否存在
        if not os.path.exists(self.embed_file_path):
            print(f"警告: 找不到预训练嵌入文件 {self.embed_file_path}")
            self.embeds = None
            return
        
        # 加载预训练嵌入
        try:
            self.embeds = self._load_pretrained_embeds()
            print(f"预训练嵌入加载完成，形状: {self.embeds.shape}")
        except Exception as e:
            print(f"加载预训练嵌入失败: {e}")
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
                if user in self._u2idx:
                    idx = self._u2idx[user]
                    embeds[idx] = np.array([float(x) for x in parts[1:]])
        
        return embeds
    
    def _create_batches(self):
        """创建数据批次"""
        print("创建数据批次...")
        
        # 训练批次
        self.train_batches = []
        for i in range(0, len(self.train_cascades), self.batch_size):
            # 确保索引有效
            end_idx = min(i + self.batch_size, len(self.train_cascades))
            
            # 确保train_intervals存在且长度匹配
            if not hasattr(self, 'train_intervals') or len(self.train_intervals) < end_idx:
                print("警告: train_intervals不存在或长度不匹配，创建默认间隔")
                self.train_intervals = [self._create_default_intervals(c) for c in self.train_cascades]
            
            batch = self._create_batch(
                self.train_cascades[i:end_idx], 
                self.train_intervals[i:end_idx]
            )
            if batch is not None:
                self.train_batches.append(batch)
        
        # 验证批次
        self.valid_batches = []
        for i in range(0, len(self.valid_cascades), self.batch_size):
            # 确保索引有效
            end_idx = min(i + self.batch_size, len(self.valid_cascades))
            
            # 确保valid_intervals存在且长度匹配
            if not hasattr(self, 'valid_intervals') or len(self.valid_intervals) < end_idx:
                print("警告: valid_intervals不存在或长度不匹配，创建默认间隔")
                self.valid_intervals = [self._create_default_intervals(c) for c in self.valid_cascades]
            
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
            
            # 确保test_intervals存在且长度匹配
            if not hasattr(self, 'test_intervals') or len(self.test_intervals) < end_idx:
                print("警告: test_intervals不存在或长度不匹配，创建默认间隔")
                self.test_intervals = [self._create_default_intervals(c) for c in self.test_cascades]
            
            batch = self._create_batch(
                self.test_cascades[i:end_idx], 
                self.test_intervals[i:end_idx]
            )
            if batch is not None:
                self.test_batches.append(batch)
        
        print(f"创建了 {len(self.train_batches)} 个训练批次, {len(self.valid_batches)} 个验证批次, {len(self.test_batches)} 个测试批次")

    def _create_batch(self, cascades, intervals):
        """创建单个批次"""
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
            
            # 动态分割点 - 使用70%作为输入，但至少保留3个节点作为目标
            split_point = max(1, min(len(cascade) - 3, int(len(cascade) * 0.7)))
            
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

    def _create_default_intervals(self, cascade):
        """创建默认的时间间隔（如果原始数据中没有时间信息）"""
        # 默认使用均匀间隔
        return [1.0] * len(cascade) 