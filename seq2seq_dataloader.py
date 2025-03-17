import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle
import scipy.sparse as sp
import gc

from seq2seq_model import normalize, sparse_mx_to_torch_sparse_tensor

"""
改进的序列到序列数据加载器:
    1. 处理级联数据和时间戳
    2. 计算时间间隔
    3. 分割序列为输入和目标
    4. 加载社交网络数据
"""
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
        
        print(f"训练集大小: {len(self.train_cascades)}")
        print(f"验证集大小: {len(self.valid_cascades)}")
        print(f"测试集大小: {len(self.test_cascades)}")
    
    def _read_cascades(self, file_path, max_len=None):
        """读取级联数据，包括时间戳，限制最大长度"""
        if max_len is None:
            max_len = self.max_seq_length
            
        cascades = []
        timestamps = []
        
        for line in open(file_path):
            if len(line.strip()) == 0:
                continue
            
            userlist = []
            timelist = []
            chunks = line.strip().split()
            
            for chunk in chunks:
                parts = chunk.split(',')
                if len(parts) >= 2:
                    user, timestamp = parts[0], float(parts[1])
                    if user in self._u2idx:
                        userlist.append(self._u2idx[user])
                        timelist.append(timestamp)
            
            # 只保留长度大于2且小于max_len的级联
            if 2 < len(userlist) <= max_len:
                cascades.append(userlist)
                timestamps.append(timelist)
        
        return cascades, timestamps
    
    def _calculate_time_intervals(self, timestamps_list):
        """计算时间间隔"""
        intervals_list = []
        
        for timestamps in timestamps_list:
            if len(timestamps) <= 1:
                intervals = [0.0]
            else:
                # 计算相邻时间戳之间的间隔
                intervals = [0.0]  # 第一个时间戳的间隔为0
                for i in range(1, len(timestamps)):
                    interval = timestamps[i] - timestamps[i-1]
                    # 归一化时间间隔，避免数值过大
                    interval = min(interval, 86400) / 86400  # 限制最大为1天，并归一化
                    intervals.append(interval)
            
            intervals_list.append(intervals)
        
        return intervals_list
    
    def _load_network_data(self):
        """加载社交网络数据"""
        # 读取边数据
        self.adj_list = [[], [], []]
        self.adj_dict = {}
        
        # 添加自环
        for i in range(self.user_size):
            self.adj_list[0].append(i)
            self.adj_list[1].append(i)
            self.adj_list[2].append(1)
            self.adj_dict[i] = [i]
        
        # 读取边
        for line in open(self.net_data_path):
            if len(line.strip()) == 0:
                continue
            
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx or nodes[1] not in self._u2idx:
                continue
            
            u1 = self._u2idx[nodes[0]]
            u2 = self._u2idx[nodes[1]]
            
            self.adj_list[0].append(u1)
            self.adj_list[1].append(u2)
            self.adj_list[2].append(1)
            
            if u1 not in self.adj_dict:
                self.adj_dict[u1] = []
            if u2 not in self.adj_dict:
                self.adj_dict[u2] = []
            
            self.adj_dict[u1].append(u2)
            self.adj_dict[u2].append(u1)
        
        # 创建稀疏邻接矩阵
        adj = sp.coo_matrix((self.adj_list[2], (self.adj_list[0], self.adj_list[1])),
                           shape=(self.user_size, self.user_size),
                           dtype=np.float32)
        
        # 归一化邻接矩阵
        adj = normalize(adj)
        
        # 转换为PyTorch稀疏张量
        self.adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
        if self.cuda:
            self.adj_tensor = self.adj_tensor.cuda()
        
        # 加载预训练嵌入（如果有）
        try:
            self.embeds = self._load_embeddings()
            print("已加载预训练嵌入")
        except:
            print("无法加载预训练嵌入")
            self.embeds = None
    
    def _load_embeddings(self):
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
        """创建训练、验证和测试批次"""
        # 计算时间间隔
        self.train_intervals = self._calculate_time_intervals(self.train_timestamps)
        self.valid_intervals = self._calculate_time_intervals(self.valid_timestamps)
        self.test_intervals = self._calculate_time_intervals(self.test_timestamps)
        
        self.train_batches = self._create_seq2seq_batches(self.train_cascades, self.train_intervals)
        self.valid_batches = self._create_seq2seq_batches(self.valid_cascades, self.valid_intervals)
        self.test_batches = self._create_seq2seq_batches(self.test_cascades, self.test_intervals)
    
    def _create_seq2seq_batches(self, cascades, intervals):
        """创建序列到序列批次"""
        # 将级联和时间间隔配对
        paired_data = list(zip(cascades, intervals))
        
        # 根据级联长度排序
        sorted_data = sorted(paired_data, key=lambda x: len(x[0]))
        
        # 创建批次
        batches = []
        current_batch = []
        
        for cascade, interval in sorted_data:
            if len(current_batch) == self.batch_size:
                batches.append(self._prepare_seq2seq_batch([item[0] for item in current_batch], 
                                                          [item[1] for item in current_batch]))
                current_batch = []
            
            current_batch.append((cascade, interval))
        
        # 添加最后一个批次
        if current_batch:
            batches.append(self._prepare_seq2seq_batch([item[0] for item in current_batch], 
                                                      [item[1] for item in current_batch]))
        
        # 打乱批次顺序
        if self.shuffle:
            random.shuffle(batches)
        
        return batches
    
    def _prepare_seq2seq_batch(self, cascades, intervals, split_ratio=None):
        """准备序列到序列批次，使用半精度浮点数减少内存使用"""
        # 使用传入的分割比例或默认值
        if split_ratio is None:
            split_ratio = self.split_ratio
        
        # 找到最长的级联
        max_len = min(max(len(c) for c in cascades), self.max_seq_length)
        
        # 创建源序列和目标序列
        src_seqs = []
        tgt_seqs = []
        src_lengths = []
        src_intervals = []
        
        for cascade, interval in zip(cascades, intervals):
            # 截断过长的序列
            if len(cascade) > self.max_seq_length:
                cascade = cascade[:self.max_seq_length]
                interval = interval[:self.max_seq_length]
                
            # 计算分割点
            split_point = max(2, int(len(cascade) * split_ratio))
            
            # 创建源序列和目标序列
            src = cascade[:split_point]
            tgt = [Constants.BOS] + cascade[split_point:] + [Constants.EOS]
            
            # 获取源序列的时间间隔
            src_interval = interval[:split_point]
            
            # 记录源序列长度
            src_lengths.append(len(src))
            
            # 填充序列
            src = src + [Constants.PAD] * (max_len - len(src))
            tgt = tgt + [Constants.PAD] * (max_len - len(tgt))
            src_interval = src_interval + [0.0] * (max_len - len(src_interval))
            
            src_seqs.append(src)
            tgt_seqs.append(tgt)
            src_intervals.append(src_interval)
        
        # 转换为张量
        src_tensor = torch.LongTensor(src_seqs)
        tgt_tensor = torch.LongTensor(tgt_seqs)
        src_lengths_tensor = torch.LongTensor(src_lengths)
        src_intervals_tensor = torch.FloatTensor(src_intervals)
        
        # 使用半精度浮点数减少内存使用
        src_intervals_tensor = src_intervals_tensor.half().float()
        
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