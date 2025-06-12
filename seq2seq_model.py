"""
Graph-Augmented Seq2Seq模型
用于社交网络信息扩散预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import Constants
import math

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

class GraphConvolution(nn.Module):
    """图卷积网络层"""
    
    def __init__(self, in_features, out_features, adj_matrix, bias=True, activation=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_matrix = adj_matrix  # 邻接矩阵（稀疏）
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.activation = activation
        self.reset_parameters()
        
    def reset_parameters(self):
        """重置参数"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x):
        """
        前向传播
        x: 节点特征矩阵 [num_nodes, in_features]
        """
        # 计算: AXW
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(self.adj_matrix, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class PositionalEncoding(nn.Module):
    """位置编码，用于序列中节点位置信息的编码"""
    
    def __init__(self, d_model, max_seq_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SocialGraphEmbedding(nn.Module):
    """社交图嵌入模块，整合节点ID嵌入和图结构信息"""
    
    def __init__(self, user_size, embed_dim, adj_tensor=None, pretrained_embeds=None, dropout=0.2):
        super(SocialGraphEmbedding, self).__init__()
        
        self.user_size = user_size
        self.embed_dim = embed_dim
        self.adj_tensor = adj_tensor  # 社交网络邻接矩阵
        
        # ID嵌入层
        self.user_embedding = nn.Embedding(user_size, embed_dim, padding_idx=Constants.PAD)
        
        # 使用预训练嵌入初始化（如果有）
        if pretrained_embeds is not None:
            self.user_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeds))
            
        # 图卷积层 - 捕获社交网络结构
        if adj_tensor is not None:
            self.gcn1 = GraphConvolution(embed_dim, embed_dim, adj_tensor)
            self.gcn2 = GraphConvolution(embed_dim, embed_dim, adj_tensor)
        else:
            self.gcn1 = None
            self.gcn2 = None
            
        # 融合层 - 结合ID嵌入和图嵌入
        self.fusion = nn.Linear(embed_dim * 2, embed_dim) if adj_tensor is not None else None
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, user_indices=None):
        """
        参数:
            user_indices: 用户索引。如果为None，则返回所有用户的嵌入
        """
        # 获取ID嵌入
        if user_indices is None:
            id_embeds = self.user_embedding.weight
        else:
            id_embeds = self.user_embedding(user_indices)
            
        # 如果没有图结构信息，直接返回ID嵌入
        if self.gcn1 is None:
            return self.dropout(id_embeds)
            
        # 计算图嵌入
        if user_indices is None:
            # 所有节点的图嵌入
            graph_embeds = self.gcn2(self.gcn1(id_embeds))
        else:
            # 获取所有节点的图嵌入，然后选择对应索引
            all_graph_embeds = self.gcn2(self.gcn1(self.user_embedding.weight))
            graph_embeds = all_graph_embeds[user_indices]
            
        # 融合ID嵌入和图嵌入
        combined = torch.cat([id_embeds, graph_embeds], dim=-1)
        fused_embeds = self.fusion(combined)
        
        return self.dropout(fused_embeds)

class AttentionLayer(nn.Module):
    """注意力层，用于解码器关注编码器输出的关键部分"""
    
    def __init__(self, hidden_size, method="general"):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        
        if method == "general":
            # general注意力需要一个变换矩阵
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == "concat":
            # concat注意力需要两个变换
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
            
    def forward(self, hidden, encoder_outputs):
        """
        计算注意力权重
        
        参数:
            hidden: [batch_size, hidden_size]
            encoder_outputs: [batch_size, seq_len, hidden_size]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # 创建注意力权重张量
        attn_energies = torch.zeros(batch_size, seq_len).to(encoder_outputs.device)
        
        # 为每个batch的每个时间步计算注意力权重
        for b in range(batch_size):
            for i in range(seq_len):
                attn_energies[b, i] = self.score(hidden[b], encoder_outputs[b, i])
                
        # Softmax归一化
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
    
    def score(self, hidden, encoder_output):
        """
        计算单个时间步的注意力分数
        
        参数:
            hidden: [hidden_size]
            encoder_output: [hidden_size]
        """
        if self.method == "dot":
            # 点积注意力
            return torch.dot(hidden, encoder_output)
        elif self.method == "general":
            # 加权点积注意力
            energy = self.attn(encoder_output)
            return torch.dot(hidden, energy)
        elif self.method == "concat":
            # 连接注意力
            energy = self.attn(torch.cat([hidden, encoder_output], dim=0))
            return torch.dot(self.v, torch.tanh(energy))

class EncoderRNN(nn.Module):
    """编码器RNN，处理输入序列，整合社交图嵌入"""
    
    def __init__(self, social_embed, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.social_embed = social_embed
        self.embed_dim = social_embed.embed_dim
        
        # 位置编码
        self.position_enc = PositionalEncoding(self.embed_dim, dropout=dropout)
        
        # GRU层 - 双向
        self.gru = nn.GRU(
            self.embed_dim + 1,  # +1 用于时间间隔特征
            hidden_size,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 输出投影层，用于将双向输出合并
        self.fc_out = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, hidden_size)
        
    def forward(self, input_seq, input_lengths, time_intervals):
        """
        参数:
            input_seq: [batch_size, seq_len]
            input_lengths: [batch_size]
            time_intervals: [batch_size, seq_len]
        """
        # 获取社交增强的节点嵌入
        embedded = self.social_embed(input_seq)  # [batch_size, seq_len, embed_dim]
        
        # 添加位置编码
        embedded = self.position_enc(embedded)
        
        # 合并时间间隔特征
        time_intervals = time_intervals.unsqueeze(-1)  # [batch_size, seq_len, 1]
        embedded = torch.cat([embedded, time_intervals], dim=-1)  # [batch_size, seq_len, embed_dim+1]
        
        # 打包序列以处理变长输入
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 通过GRU传递
        outputs, hidden = self.gru(packed)
        
        # 解包序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # 如果是双向GRU，合并前向和后向状态
        if self.bidirectional:
            # 合并双向输出 [batch_size, seq_len, hidden_size*2]
            outputs = self.fc_out(outputs)
            
            # 合并最终隐藏状态
            # 前向和后向最后一层的隐藏状态
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden = torch.tanh(self.fc_out(torch.cat([hidden_forward, hidden_backward], dim=1)))
            hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        return outputs, hidden

class SocialAwareDecoderRNN(nn.Module):
    """社交感知解码器RNN，使用注意力机制和社交网络约束"""
    
    def __init__(self, social_embed, hidden_size, output_size, adj_matrix=None, n_layers=1, dropout=0.1):
        super(SocialAwareDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size  # 用户总数
        self.n_layers = n_layers
        self.social_embed = social_embed
        self.embed_dim = social_embed.embed_dim
        self.adj_matrix = adj_matrix  # 邻接矩阵，用于社交约束
        
        # 注意力层
        self.attention = AttentionLayer(hidden_size)
        
        # GRU层
        self.gru = nn.GRU(
            self.embed_dim + hidden_size,  # 嵌入 + 上下文向量
            hidden_size,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层 - 预测下一个节点
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_step, last_hidden, encoder_outputs, last_node=None):
        """
        参数:
            input_step: [batch_size] - 当前步骤的输入节点
            last_hidden: [n_layers, batch_size, hidden_size] - 上一步的隐藏状态
            encoder_outputs: [batch_size, seq_len, hidden_size] - 编码器输出
            last_node: [batch_size] - 上一个预测的节点，用于社交约束（可选）
        """
        # 获取当前步骤的嵌入
        embedded = self.social_embed(input_step)  # [batch_size, embed_dim]
        embedded = self.dropout(embedded)
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 计算注意力权重
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)  # [batch_size, 1, seq_len]
        
        # 计算上下文向量
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_size]
        
        # 合并嵌入和上下文
        rnn_input = torch.cat([embedded, context], dim=2)  # [batch_size, 1, embed_dim+hidden_size]
        
        # GRU计算
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)  # [batch_size, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        # 预测下一个节点
        output = torch.cat([output, context], dim=1)  # [batch_size, hidden_size*2]
        output = self.out(output)  # [batch_size, output_size]
        
        # 如果提供了上一个节点且有邻接矩阵，应用社交网络约束
        if last_node is not None and self.adj_matrix is not None:
            # 为每个batch获取邻居节点的掩码
            batch_size = last_node.size(0)
            device = output.device
            
            # 创建掩码，初始化为全0（屏蔽所有节点）
            mask = torch.zeros(batch_size, self.output_size).to(device)
            
            # 对每个batch单独处理
            for i in range(batch_size):
                node_idx = last_node[i].item()
                
                # 如果是特殊标记，不应用约束
                if node_idx < 4:  # 小于4的是特殊标记
                    mask[i, :] = 1.0  # 不屏蔽任何节点
                else:
                    # 获取当前节点的邻居（邻接矩阵中的非零元素）
                    neighbors = self.adj_matrix[node_idx].coalesce().indices()[1]
                    mask[i, neighbors] = 1.0  # 只允许邻居节点
                    
                    # 特殊标记总是可用
                    mask[i, :4] = 1.0
            
            # 应用掩码：将非邻居节点的概率设为非常小的值
            output = output * mask + (1 - mask) * -1e9
        
        return output, hidden, attn_weights

class GraphAugmentedSeq2Seq(nn.Module):
    """图增强的序列到序列模型，用于社交网络信息扩散预测"""
    
    def __init__(self, user_size, embed_dim, hidden_size, n_layers=2, dropout=0.1, 
                 adj_tensor=None, pretrained_embeds=None, bidirectional_encoder=True):
        super(GraphAugmentedSeq2Seq, self).__init__()
        
        # 社交图嵌入
        self.social_embed = SocialGraphEmbedding(
            user_size, 
            embed_dim, 
            adj_tensor, 
            pretrained_embeds,
            dropout
        )
        
        # 编码器
        self.encoder = EncoderRNN(
            self.social_embed,
            hidden_size,
            n_layers,
            dropout,
            bidirectional_encoder
        )
        
        # 解码器
        self.decoder = SocialAwareDecoderRNN(
            self.social_embed,
            hidden_size,
            user_size,
            adj_tensor,
            n_layers,
            dropout
        )
        
    def forward(self, src, src_lengths, time_intervals, tgt, teacher_forcing_ratio=0.5):
        """
        参数:
            src: [batch_size, src_len] - 输入序列
            src_lengths: [batch_size] - 输入序列真实长度
            time_intervals: [batch_size, src_len] - 时间间隔
            tgt: [batch_size, tgt_len] - 目标序列
            teacher_forcing_ratio: 使用教师强制的概率
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.output_size
        
        # 输出张量初始化
        outputs = torch.zeros(batch_size, tgt_len-1, tgt_vocab_size).to(src.device)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src, src_lengths, time_intervals)
        
        # 第一个解码输入是BOS标记
        decoder_input = tgt[:, 0]  # [batch_size]
        
        # 用于社交约束的上一个节点
        last_node = None
        
        # 解码
        for t in range(1, tgt_len):
            # 解码一个步骤
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, last_node)
            outputs[:, t-1, :] = output
            
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # 获取最可能的单词
            _, topi = output.topk(1)
            decoder_input = tgt[:, t] if teacher_force else topi.squeeze(-1)
            
            # 更新上一个节点（用于社交约束）
            last_node = decoder_input
        
        return outputs
    
    def predict(self, src, src_lengths, time_intervals, max_length=3):
        """
        预测给定序列的后续节点
        
        参数:
            src: [batch_size, src_len]
            src_lengths: [batch_size]
            time_intervals: [batch_size, src_len]
            max_length: 预测的最大长度
        """
        batch_size = src.size(0)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src, src_lengths, time_intervals)
        
        # 开始标记
        decoder_input = torch.full((batch_size,), Constants.BOS, device=src.device)
        
        # 用于社交约束的上一个节点
        last_node = None
        
        # 存储预测
        predictions = []
        attentions = []
        
        # 逐步解码
        for _ in range(max_length):
            # 解码一个步骤
            output, hidden, attn_weights = self.decoder(decoder_input, hidden, encoder_outputs, last_node)
            
            # 获取最可能的用户
            _, topi = output.topk(1)
            decoder_input = topi.squeeze(-1)
            
            # 更新上一个节点（用于社交约束）
            last_node = decoder_input
            
            # 添加到预测中
            predictions.append(decoder_input.detach())
            attentions.append(attn_weights.detach())
        
        # 堆叠预测结果
        return torch.stack(predictions, dim=1), attentions  # [batch_size, max_length] 