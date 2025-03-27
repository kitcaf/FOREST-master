import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
import Constants
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""
改进的序列到序列模型:
    1. 时间感知嵌入
    2. Transformer编码器
    3. 图注意力网络(GAT)
    4. 多头自注意力机制
    5. 时间间隔编码
"""

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_seq_length=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不作为模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeIntervalEncoding(nn.Module):
    """时间间隔编码模块"""
    def __init__(self, d_model, dropout=0.1):
        super(TimeIntervalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.time_embedding = nn.Linear(1, d_model)
        
    def forward(self, x, time_intervals):
        # x: [batch_size, seq_len, d_model]
        # time_intervals: [batch_size, seq_len]
        
        # 将时间间隔转换为嵌入
        time_intervals = time_intervals.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        time_embed = self.time_embedding(time_intervals)  # [batch_size, seq_len, d_model]
        
        # 将时间嵌入添加到输入中
        x = x + time_embed
        return self.dropout(x)

class GraphAttentionLayer(nn.Module):
    """改进的图注意力层"""
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # 定义可学习参数
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # 激活函数和dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, h, adj):
        """
        h: 节点特征 [N, in_features]
        adj: 邻接矩阵 [N, N]
        """
        # 线性变换
        Wh = self.W(h)  # [N, out_features]
        
        # 计算注意力系数
        N = Wh.size(0)
        
        # 创建所有节点对的特征组合
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        
        # 计算注意力系数
        e = torch.matmul(a_input, self.a).squeeze(2)
        e = self.leakyrelu(e)
        
        # 掩码注意力系数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # 应用注意力系数
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % n_heads == 0
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # 分别为Q, K, V创建线性层
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性变换
        Q = self.fc_q(query)  # [batch_size, query_len, hidden_dim]
        K = self.fc_k(key)    # [batch_size, key_len, hidden_dim]
        V = self.fc_v(value)  # [batch_size, value_len, hidden_dim]
        
        # 分离头
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, n_heads, query_len, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, n_heads, key_len, head_dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, n_heads, value_len, head_dim]
        
        # 确保设备一致
        if hasattr(self, 'scale') and isinstance(self.scale, torch.Tensor):
            self.scale = self.scale.to(query.device)
        else:
            self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(query.device)
        
        # 缩放点积注意力
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, query_len, key_len]
        
        # 应用掩码（如果有）
        if mask is not None:
            # 调整掩码维度以匹配energy
            if mask.dim() == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # 注意力权重
        attention = F.softmax(energy, dim=-1)  # [batch_size, n_heads, query_len, key_len]
        attention = self.dropout(attention)
        
        # 应用注意力权重
        x = torch.matmul(attention, V)  # [batch_size, n_heads, query_len, head_dim]
        
        # 合并头
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
        x = x.view(batch_size, -1, self.hidden_dim)  # [batch_size, query_len, hidden_dim]
        
        # 最终线性层
        x = self.fc_o(x)  # [batch_size, query_len, hidden_dim]
        
        return x

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout)
        
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hidden_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # src: [batch_size, src_len, hidden_dim]
        # src_mask: [batch_size, src_len]
        
        # 自注意力
        _src = self.self_attention(src, src, src, src_mask)
        
        # 残差连接和层归一化
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        # 前馈网络
        _src = self.positionwise_feedforward(src)
        
        # 残差连接和层归一化
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        return src

class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttention(hidden_dim, n_heads, dropout)
        
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hidden_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_src, tgt_mask=None, src_mask=None):
        # tgt: [batch_size, tgt_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # tgt_mask: [batch_size, tgt_len]
        # src_mask: [batch_size, src_len]
        
        # 自注意力
        _tgt = self.self_attention(tgt, tgt, tgt, tgt_mask)
        
        # 残差连接和层归一化
        tgt = self.self_attn_layer_norm(tgt + self.dropout(_tgt))
        
        # 编码器-解码器注意力
        _tgt = self.encoder_attention(tgt, enc_src, enc_src, src_mask)
        
        # 残差连接和层归一化
        tgt = self.enc_attn_layer_norm(tgt + self.dropout(_tgt))
        
        # 前馈网络
        _tgt = self.positionwise_feedforward(tgt)
        
        # 残差连接和层归一化
        tgt = self.ff_layer_norm(tgt + self.dropout(_tgt))
        
        return tgt

class Encoder(nn.Module):
    """改进的编码器模块"""
    def __init__(self, user_size, embed_dim, hidden_dim, n_layers=2, n_heads=8, pf_dim=512, dropout=0.1, 
                 use_network=False, adj=None, net_dict=None, max_seq_length=1000):
        super(Encoder, self).__init__()
        
        self.user_size = user_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_network = use_network
        self.adj = adj
        
        # 用户嵌入
        self.user_embedding = nn.Embedding(user_size, embed_dim, padding_idx=Constants.PAD)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_length, dropout)
        
        # 时间间隔编码
        self.time_encoder = TimeIntervalEncoding(embed_dim, dropout)
        
        # 图注意力网络
        if use_network and adj is not None:
            self.gat = GraphAttentionLayer(embed_dim, embed_dim, dropout=dropout)
            # 添加第二层GAT以捕获更复杂的图结构
            self.gat2 = GraphAttentionLayer(embed_dim, embed_dim, dropout=dropout)
        
        # 投影层，将嵌入维度映射到隐藏维度
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        
        # 编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, n_heads, pf_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lengths, time_intervals=None):
        # src: [batch_size, src_len]
        # src_lengths: [batch_size]
        # time_intervals: [batch_size, src_len] - 可选的时间间隔信息
        
        batch_size, src_len = src.shape
        
        # 创建掩码
        src_mask = (src != Constants.PAD).float()  # [batch_size, src_len]
        
        # 基本嵌入
        embedded = self.user_embedding(src)  # [batch_size, src_len, embed_dim]
        
        # 应用位置编码
        embedded = self.pos_encoder(embedded)
        
        # 应用时间间隔编码（如果提供）
        if time_intervals is not None:
            embedded = self.time_encoder(embedded, time_intervals)
        
        # 应用图注意力网络（如果使用社交网络）
        if self.use_network and self.adj is not None:
            # 保存原始嵌入
            original_shape = embedded.shape
            
            # 重塑为[batch_size * src_len, embed_dim]以应用GAT
            flat_embedded = embedded.view(-1, self.embed_dim)
            
            # 应用第一层GAT
            gat_output = self.gat(flat_embedded, self.adj)
            
            # 应用第二层GAT以捕获更复杂的图结构
            gat_output = self.gat2(gat_output, self.adj)
            
            # 重塑回原始形状
            gat_output = gat_output.view(original_shape)
            
            # 残差连接
            embedded = embedded + gat_output
        
        # 投影到隐藏维度
        src = self.input_projection(embedded)  # [batch_size, src_len, hidden_dim]
        
        # 应用编码器层
        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src, embedded

class Decoder(nn.Module):
    """解码器模块"""
    def __init__(self, user_size, embed_dim, hidden_dim, n_layers=2, n_heads=8, pf_dim=512, dropout=0.1, max_seq_length=1000):
        super(Decoder, self).__init__()
        
        self.user_size = user_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 用户嵌入
        self.user_embedding = nn.Embedding(user_size, embed_dim, padding_idx=Constants.PAD)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_length, dropout)
        
        # 投影层，将嵌入维度映射到隐藏维度
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, n_heads, pf_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, user_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_src, src_mask=None):
        # tgt: [batch_size, tgt_len]
        # enc_src: [batch_size, src_len, hidden_dim]
        # src_mask: [batch_size, src_len]
        
        batch_size, tgt_len = tgt.shape
        
        # 创建掩码
        tgt_mask = (tgt != Constants.PAD).float()  # [batch_size, tgt_len]
        
        # 创建后续掩码（防止看到未来的token）
        subsequent_mask = torch.triu(
            torch.ones((tgt_len, tgt_len), device=tgt.device) * float('-inf'),
            diagonal=1
        )
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, tgt_len, tgt_len]
        
        # 基本嵌入
        embedded = self.user_embedding(tgt)  # [batch_size, tgt_len, embed_dim]
        
        # 应用位置编码
        embedded = self.pos_encoder(embedded)
        
        # 投影到隐藏维度
        tgt = self.input_projection(embedded)  # [batch_size, tgt_len, hidden_dim]
        
        # 应用解码器层
        for layer in self.layers:
            tgt = layer(tgt, enc_src, tgt_mask, src_mask)
        
        # 输出层
        output = self.fc_out(tgt)  # [batch_size, tgt_len, user_size]
        
        return output

class ImprovedSeq2SeqModel(nn.Module):
    """改进的序列到序列模型"""
    def __init__(self, user_size, embed_dim, hidden_dim, n_layers=2, n_heads=8, pf_dim=512, dropout=0.1, 
                 use_network=False, adj=None, net_dict=None, teacher_forcing_ratio=0.5, max_seq_length=1000):
        super(ImprovedSeq2SeqModel, self).__init__()
        
        # 确保adj是稀疏张量
        if adj is not None and not isinstance(adj, torch.sparse.FloatTensor):
            print("警告：邻接矩阵不是稀疏张量，性能可能受影响")
        
        self.encoder = Encoder(
            user_size, embed_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout, 
            use_network, adj, net_dict, max_seq_length
        )
        
        self.decoder = Decoder(
            user_size, embed_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout, max_seq_length
        )
        
        self.user_size = user_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_dim = hidden_dim
        self.use_network = use_network
    
    def create_mask(self, src):
        """创建源序列掩码"""
        return (src != Constants.PAD).float()
    
    def forward(self, src, src_lengths, tgt, time_intervals=None, teacher_forcing_ratio=None):
        """前向传播"""
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio
            
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # 创建掩码
        src_mask = self.create_mask(src)  # [batch_size, src_len]
        
        # 编码
        enc_src, _ = self.encoder(src, src_lengths, time_intervals)  # [batch_size, src_len, hidden_dim]
        
        # 存储预测结果
        outputs = torch.zeros(batch_size, tgt_len, self.user_size).to(src.device)
        
        # 初始化解码器输入
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # 逐步解码
        for t in range(1, tgt_len):
            # 解码一步
            output = self.decoder(decoder_input, enc_src, src_mask)  # [batch_size, t, user_size]
            
            # 存储当前步的预测
            outputs[:, t, :] = output[:, -1, :]
            
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # 获取下一步的输入
            top1 = output[:, -1, :].argmax(1).unsqueeze(1)  # [batch_size, 1]
            
            if teacher_force:
                decoder_input = torch.cat([decoder_input, tgt[:, t].unsqueeze(1)], dim=1)
            else:
                decoder_input = torch.cat([decoder_input, top1], dim=1)
            
        return outputs
    
    def generate(self, src, src_lengths, max_len=3, time_intervals=None, start_tokens=None):
        """生成序列，固定生成3个节点"""
        batch_size = src.size(0)
        
        # 创建掩码
        src_mask = self.create_mask(src)  # [batch_size, src_len]
        
        # 编码
        enc_src, _ = self.encoder(src, src_lengths, time_intervals)  # [batch_size, src_len, hidden_dim]
        
        # 初始化解码器输入
        if start_tokens is None:
            decoder_input = torch.tensor([[Constants.BOS]] * batch_size).to(src.device)  # [batch_size, 1]
        else:
            decoder_input = start_tokens.unsqueeze(1)  # [batch_size, 1]
        
        # 存储生成的序列
        generated_seq = torch.zeros(batch_size, max_len, dtype=torch.long).to(src.device)
        generated_probs = torch.zeros(batch_size, max_len, self.user_size).to(src.device)
        
        # 已生成的用户集合（避免重复）
        generated_users = [set() for _ in range(batch_size)]
        
        # 逐步生成，固定生成3个节点
        for t in range(max_len):
            # 解码一步
            output = self.decoder(decoder_input, enc_src, src_mask)  # [batch_size, t+1, user_size]
            
            # 获取最后一步的输出
            step_output = output[:, -1, :]  # [batch_size, user_size]
            
            # 应用掩码防止生成已经出现过的用户
            for i in range(batch_size):
                for j in range(decoder_input.size(1)):
                    user_id = decoder_input[i, j].item()
                    if user_id not in [Constants.PAD, Constants.EOS, Constants.BOS]:
                        generated_users[i].add(user_id)
                
                for user_id in generated_users[i]:
                    step_output[i, user_id] = float('-inf')
            
            # 存储概率分布
            generated_probs[:, t, :] = F.softmax(step_output, dim=1)
            
            # 采样下一个用户
            top1 = step_output.argmax(1)  # 使用贪婪搜索确保最佳预测
            
            # 存储生成的用户
            generated_seq[:, t] = top1
            
            # 更新解码器输入
            decoder_input = torch.cat([decoder_input, top1.unsqueeze(1)], dim=1)  # [batch_size, t+2]
            
        return generated_seq, generated_probs 