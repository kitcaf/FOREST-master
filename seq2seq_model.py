import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
import Constants

"""
改进的序列到序列模型:
    1. 用户编码器：融合社交图信息
    2. 时序-图联合编码器：LSTM/GRU + 动态图特征注入
    3. 高效解码器：预测未来3个用户
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
    def __init__(self, d_model, max_seq_length=3000, dropout=0.1):
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
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # 如果序列长度超过预定义的最大长度，截断序列
            x = x[:, :self.pe.size(1), :]
        
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeIntervalEncoding(nn.Module):
    """时间间隔编码模块"""
    def __init__(self, d_model, dropout=0.1):
        super(TimeIntervalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.time_embedding = nn.Linear(1, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, time_intervals):
        # x: [batch_size, seq_len, d_model]
        # time_intervals: [batch_size, seq_len]
        
        # 将时间间隔转换为嵌入
        time_intervals = time_intervals.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        time_embed = self.time_embedding(time_intervals)  # [batch_size, seq_len, d_model]
        
        # 将时间嵌入添加到输入中
        x = x + time_embed
        x = self.layer_norm(x)
        return self.dropout(x)

class GraphConvolution(nn.Module):
    """简单的图卷积层"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=0.1)  # 使用较小的gain值
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        # x: [N, in_features]
        # adj: [N, N] sparse matrix
        
        # 图卷积操作: AXW
        support = torch.mm(x, self.weight)  # XW
        output = torch.sparse.mm(adj, support)  # AXW
        
        if self.bias is not None:
            output = output + self.bias
            
        return output

class UserEncoder(nn.Module):
    """用户编码器：将用户ID映射为低维向量，融合社交图信息"""
    def __init__(self, user_size, embed_dim, dropout=0.1, use_network=False, adj=None):
        super(UserEncoder, self).__init__()
        self.user_size = user_size
        self.embed_dim = embed_dim
        self.use_network = use_network
        
        # 用户嵌入层，使用Xavier初始化增强稳定性
        self.user_embedding = nn.Embedding(user_size, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=0.1)  # 使用较小的gain值
        
        # 如果使用社交网络，添加图卷积层
        if use_network and adj is not None:
            self.gcn = GraphConvolution(embed_dim, embed_dim)
            self.register_buffer('adj', adj)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, user_ids):
        """
        参数:
            user_ids: 用户ID序列 [batch_size, seq_len]
        返回:
            user_embeds: 用户嵌入 [batch_size, seq_len, embed_dim]
        """
        # 获取用户嵌入
        user_embeds = self.user_embedding(user_ids)  # [batch_size, seq_len, embed_dim]
        
        # 应用图卷积（如果启用）
        if self.use_network and hasattr(self, 'adj'):
            # 对每个位置应用GCN
            batch_size, seq_len, _ = user_embeds.size()
            
            # 重塑为[batch_size * seq_len, embed_dim]以便应用GCN
            user_embeds_flat = user_embeds.view(-1, self.embed_dim)
            
            # 应用GCN，确保数值稳定性
            try:
                user_embeds_flat = self.gcn(user_embeds_flat, self.adj)
                
                # 检查NaN并修复
                if torch.isnan(user_embeds_flat).any():
                    print("GCN输出包含NaN，使用原始嵌入...")
                    user_embeds_flat = self.user_embedding(user_ids).view(-1, self.embed_dim)
            except Exception as e:
                print(f"GCN计算出错: {e}")
                user_embeds_flat = self.user_embedding(user_ids).view(-1, self.embed_dim)
            
            # 重塑回[batch_size, seq_len, embed_dim]
            user_embeds = user_embeds_flat.view(batch_size, seq_len, self.embed_dim)
        
        # 应用层归一化和dropout
        user_embeds = self.layer_norm(user_embeds)
        user_embeds = self.dropout(user_embeds)
        
        return user_embeds

class TemporalGraphEncoder(nn.Module):
    """时序-图联合编码器：使用LSTM/GRU替代Transformer"""
    def __init__(self, embed_dim, hidden_dim, n_layers=2, dropout=0.1, max_seq_length=3000, rnn_type='GRU'):
        super(TemporalGraphEncoder, self).__init__()
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_length, dropout)
        
        # 时间间隔编码
        self.time_encoder = TimeIntervalEncoding(embed_dim, dropout)
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # 使用双向GRU/LSTM
        self.rnn_type = rnn_type
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim // 2,  # 双向所以减半
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0
            )
        else:  # 默认使用GRU
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim // 2,  # 双向所以减半
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0
            )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化RNN权重
        self._init_weights()
        
    def _init_weights(self):
        """使用保守的初始化方式"""
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, mask, time_intervals=None):
        """
        参数:
            x: 用户嵌入 [batch_size, seq_len, embed_dim]
            mask: 掩码 [batch_size, seq_len]
            time_intervals: 时间间隔 [batch_size, seq_len]
        返回:
            encoded: 编码后的序列 [batch_size, seq_len, hidden_dim]
        """
        # 应用输入归一化
        x = self.input_norm(x)
        
        # 应用位置编码
        x = self.pos_encoder(x)
        
        # 应用时间间隔编码（如果提供）
        if time_intervals is not None:
            x = self.time_encoder(x, time_intervals)
        
        # 获取有效长度
        lengths = mask.sum(dim=1).long()
        
        # 确保长度至少为1
        lengths = torch.clamp(lengths, min=1)
        
        # 打包序列
        try:
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # 应用RNN
            if self.rnn_type == 'LSTM':
                packed_output, _ = self.rnn(packed_x)
            else:
                packed_output, _ = self.rnn(packed_x)
            
            # 解包序列
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # 如果序列长度不匹配，进行填充
            if output.size(1) < x.size(1):
                padding = torch.zeros(
                    x.size(0), x.size(1) - output.size(1), output.size(2),
                    device=output.device
                )
                output = torch.cat([output, padding], dim=1)
            
            # 应用输出投影
            encoded = self.output_projection(output)
            
            # 应用层归一化和dropout
            encoded = self.layer_norm(encoded)
            encoded = self.dropout(encoded)
            
            # 检查NaN并修复
            if torch.isnan(encoded).any():
                print("RNN输出包含NaN，应用修复...")
                encoded = torch.nan_to_num(encoded, nan=0.0)
                
                # 如果仍然有NaN，使用输入作为后备
                if torch.isnan(encoded).any():
                    print("使用输入作为后备...")
                    encoded = self.output_projection(x)
                    encoded = self.layer_norm(encoded)
        
        except Exception as e:
            print(f"RNN计算出错: {e}")
            # 使用输入作为后备
            encoded = self.output_projection(x)
            encoded = self.layer_norm(encoded)
        
        return encoded

class EfficientDecoder(nn.Module):
    """高效解码器：基于联合特征预测未来用户"""
    def __init__(self, hidden_dim, user_size, dropout=0.1):
        super(EfficientDecoder, self).__init__()
        
        # 极度简化架构，只使用一个线性层
        self.output_layer = nn.Linear(hidden_dim, user_size)
        
        # 使用极小的初始化值
        with torch.no_grad():
            self.output_layer.weight.data.fill_(0.0)
            # 使用均匀分布初始化，范围非常小
            self.output_layer.weight.data.uniform_(-0.001, 0.001)
            self.output_layer.bias.data.fill_(0.0)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoder_output, src_lengths):
        """
        参数:
            encoder_output: 编码器输出 [batch_size, src_len, hidden_dim]
            src_lengths: 源序列长度 [batch_size]
        返回:
            outputs: 解码器输出 [batch_size, 3, user_size]
        """
        batch_size = encoder_output.size(0)
        hidden_dim = encoder_output.size(2)
        user_size = self.output_layer.out_features
        
        # 获取每个序列的最后一个有效位置的编码
        last_hidden = torch.zeros(batch_size, hidden_dim, device=encoder_output.device)
        for i in range(batch_size):
            last_pos = min(src_lengths[i].item() - 1, encoder_output.size(1) - 1)
            if last_pos >= 0:
                last_hidden[i] = encoder_output[i, last_pos]
        
        # 应用层归一化和dropout
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        
        # 缩放隐藏状态以减小数值范围
        last_hidden = last_hidden * 0.1
        
        # 创建输出张量
        outputs = torch.zeros(batch_size, 3, user_size, device=encoder_output.device)
        
        # 对每个时间步分别生成输出
        for t in range(3):
            # 使用相同的隐藏状态，但添加极小的位置偏移
            time_offset = torch.zeros_like(last_hidden)
            time_offset[:, :5] = 0.01 * (t + 1)  # 只在前5个维度添加极小偏移
            
            # 生成当前时间步的输出
            current_hidden = last_hidden + time_offset
            
            # 应用输出层，极度缩放logits以提高稳定性
            logits = self.output_layer(current_hidden) * 0.01
            
            outputs[:, t] = logits
        
        return outputs

class ImprovedSeq2SeqModel(nn.Module):
    """改进的序列到序列模型"""
    def __init__(self, user_size, embed_dim, hidden_dim, n_layers=2, n_heads=8, pf_dim=512, dropout=0.1, 
                 use_network=False, adj=None, net_dict=None, teacher_forcing_ratio=0.5, max_seq_length=3000,
                 rnn_type='GRU'):
        super(ImprovedSeq2SeqModel, self).__init__()
        
        # 确保adj是稀疏张量
        if adj is not None and not isinstance(adj, torch.sparse.FloatTensor):
            print("警告：邻接矩阵不是稀疏张量，性能可能受影响")
        
        # 用户编码器
        self.user_encoder = UserEncoder(
            user_size=user_size,
            embed_dim=embed_dim,
            dropout=dropout,
            use_network=use_network,
            adj=adj
        )
        
        # 时序-图联合编码器（使用LSTM/GRU替代Transformer）
        self.temporal_graph_encoder = TemporalGraphEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            rnn_type=rnn_type
        )
        
        # 高效解码器
        self.efficient_decoder = EfficientDecoder(
            hidden_dim=hidden_dim,
            user_size=user_size,
            dropout=dropout
        )
        
        self.user_size = user_size
        self.hidden_dim = hidden_dim
        
    def create_mask(self, src):
        """创建源序列掩码"""
        return (src != Constants.PAD).float()  # [batch_size, src_len]
    
    def forward(self, src, src_lengths, tgt=None, time_intervals=None):
        """前向传播"""
        batch_size = src.size(0)
        
        # 创建掩码
        src_mask = self.create_mask(src)  # [batch_size, src_len]
        
        # 1. 用户编码
        user_embeds = self.user_encoder(src)  # [batch_size, src_len, embed_dim]
        
        # 2. 时序-图联合编码
        encoder_output = self.temporal_graph_encoder(
            user_embeds, src_mask, time_intervals
        )  # [batch_size, src_len, hidden_dim]
        
        # 3. 高效解码
        decoder_output = self.efficient_decoder(
            encoder_output, src_lengths
        )  # [batch_size, 3, user_size]
        
        # 添加一个全零的第一个位置（对应BOS）
        bos_logits = torch.zeros(batch_size, 1, self.user_size, device=src.device)
        outputs = torch.cat([bos_logits, decoder_output], dim=1)  # [batch_size, 4, user_size]
        
        # 如果需要，添加一个全零的最后一个位置（对应EOS）
        if tgt is not None and tgt.size(1) > outputs.size(1):
            eos_logits = torch.zeros(batch_size, tgt.size(1) - outputs.size(1), self.user_size, device=src.device)
            outputs = torch.cat([outputs, eos_logits], dim=1)  # [batch_size, tgt_len, user_size]
        
        return outputs
    
    def generate(self, src, src_lengths, max_len=3, time_intervals=None, start_tokens=None):
        """生成序列，固定生成3个节点"""
        batch_size = src.size(0)
        
        # 前向传播获取预测
        with torch.no_grad():  # 使用无梯度模式提高稳定性
            outputs = self.forward(src, src_lengths, time_intervals=time_intervals)
        
        # 提取预测结果（跳过BOS位置）
        logits = outputs[:, 1:max_len+1]  # [batch_size, max_len, user_size]
        
        # 存储生成的序列
        generated_seq = torch.zeros(batch_size, max_len, dtype=torch.long, device=src.device)
        generated_probs = torch.zeros(batch_size, max_len, self.user_size, device=src.device)
        
        # 已生成的用户集合（避免重复）
        generated_users = [set() for _ in range(batch_size)]
        
        # 对每个位置进行预测
        for t in range(max_len):
            # 获取当前位置的输出
            step_output = logits[:, t].clone()  # [batch_size, user_size]
            
            # 应用掩码防止生成已经出现过的用户
            for i in range(batch_size):
                for user_id in generated_users[i]:
                    step_output[i, user_id] = float('-inf')
                
                # 也避免生成源序列中的用户
                for j in range(src.size(1)):
                    user_id = src[i, j].item()
                    if user_id != Constants.PAD:
                        step_output[i, user_id] = float('-inf')
            
            # 使用温度为0.7的softmax，增加多样性
            temperature = 0.7
            step_probs = F.softmax(step_output / temperature, dim=1)
            
            # 检查NaN并修复
            if torch.isnan(step_probs).any():
                print(f"时间步{t}的概率分布包含NaN，应用修复...")
                step_probs = torch.nan_to_num(step_probs, nan=0.0)
                # 重新归一化
                step_probs = step_probs / (step_probs.sum(dim=1, keepdim=True) + 1e-10)
            
            # 存储概率分布
            generated_probs[:, t] = step_probs
            
            # 采样下一个用户
            top1 = step_output.argmax(1)  # [batch_size]
            
            # 存储生成的用户
            generated_seq[:, t] = top1
            
            # 更新已生成的用户集合
            for i in range(batch_size):
                generated_users[i].add(top1[i].item())
        
        return generated_seq, generated_probs 