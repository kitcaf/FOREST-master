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
        """读取级联数据"""
        cascades = []
        timestamps = []
        
        for line in open(file_path):
            if len(line.strip()) == 0:
                continue
                
            chunks = line.strip().split()
            cascade = []
            timestamp = []
            
            for chunk in chunks:
                user, time = chunk.split(',')
                
                # 将用户名转换为ID
                if user in self._u2idx:
                    user_id = self._u2idx[user]
                else:
                    user_id = self._u2idx['<unk>']
                
                # 将时间转换为浮点数
                time = float(time)
                
                cascade.append(user_id)
                timestamp.append(time)
            
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
            interval = [0.0]  # 第一个用户的时间间隔为0
            
            for i in range(1, len(ts)):
                interval.append(ts[i] - ts[i-1])
            
            intervals.append(interval)
        
        return intervals
    
    def _load_network_data(self):
        """加载社交网络数据"""
        print("加载社交网络数据...")
        
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
        
        # 过滤掉长度小于等于3的级联
        paired_data = [pair for pair in paired_data if len(pair[0]) > 3]
        
        if not paired_data:
            return []
        
        # 根据级联长度排序
        sorted_data = sorted(paired_data, key=lambda x: len(x[0]))
        
        # 创建批次
        batches = []
        current_batch = []
        
        for cascade, interval in sorted_data:
            if len(current_batch) == self.batch_size:
                batch = self._prepare_seq2seq_batch([item[0] for item in current_batch], 
                                                  [item[1] for item in current_batch])
                if batch is not None:
                    batches.append(batch)
                current_batch = []
            
            current_batch.append((cascade, interval))
        
        # 添加最后一个批次
        if current_batch:
            batch = self._prepare_seq2seq_batch([item[0] for item in current_batch], 
                                              [item[1] for item in current_batch])
            if batch is not None:
                batches.append(batch)
        
        # 打乱批次顺序
        if self.shuffle:
            random.shuffle(batches)
        
        return batches
    
    def _prepare_seq2seq_batch(self, cascades, intervals, split_ratio=None):
        """准备序列到序列批次，输入是原序列长度-3，输出是后面3个节点"""
        # 使用传入的分割比例或默认值
        if split_ratio is None:
            split_ratio = self.split_ratio
        
        # 创建源序列和目标序列
        src_seqs = []
        tgt_seqs = []
        src_lengths = []
        src_intervals = []
        
        for cascade, interval in zip(cascades, intervals):
            # 只处理长度大于3的级联
            if len(cascade) <= 3:
                continue
            
            # 计算分割点：取除最后3个节点外的所有节点作为输入
            split_point = len(cascade) - 3
            
            # 创建源序列和目标序列
            src = cascade[:split_point]
            tgt = [Constants.BOS] + cascade[split_point:] + [Constants.EOS]  # 最后3个节点加上BOS和EOS
            
            # 获取源序列的时间间隔
            src_interval = interval[:split_point]
            
            # 记录源序列长度
            src_lengths.append(len(src))
            
            src_seqs.append(src)
            tgt_seqs.append(tgt)
            src_intervals.append(src_interval)
        
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