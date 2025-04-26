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

class PositionalEncoding(nn.Module):
    """位置编码"""
    
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

class IntentAwareSelfGating(nn.Module):
    """意图感知自门控机制，从DisenIDP中参考"""
    
    def __init__(self, input_dim, intent_type):
        super(IntentAwareSelfGating, self).__init__()
        self.intent_type = intent_type  # I (兴趣) 或 D (依赖)
        self.W = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        """
        参数:
            X: [batch_size, seq_len, input_dim] 或 [user_size, input_dim]
        """
        gate = self.sigmoid(self.W(X))
        return X * gate

class UserEmbeddingEnhancement(nn.Module):
    """用户嵌入增强模块，结合DisenIDP和MS-HGAT的思想"""
    
    def __init__(self, user_size, embed_dim, adj_tensor=None, pretrained_embeds=None):
        super(UserEmbeddingEnhancement, self).__init__()
        
        self.user_size = user_size
        self.embed_dim = embed_dim
        self.adj_tensor = adj_tensor  # 社交网络邻接矩阵
        
        # 用户嵌入层
        if pretrained_embeds is not None:
            self.user_embeds = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeds),
                padding_idx=Constants.PAD,
                freeze=False
            )
        else:
            self.user_embeds = nn.Embedding(
                user_size, 
                embed_dim,
                padding_idx=Constants.PAD
            )
        
        # DisenIDP的意图感知门控
        self.interest_gate = IntentAwareSelfGating(embed_dim, "I")
        self.dependency_gate = IntentAwareSelfGating(embed_dim, "D")
        
        # GCN层处理社交网络数据，来自MS-HGAT
        self.gcn_layer = nn.Linear(embed_dim, embed_dim)
        self.gcn_act = nn.ReLU()
        
        # 融合不同类型的表示
        self.fusion_layer = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, user_indices=None):
        """
        参数:
            user_indices: 用户索引。如果为None，则计算所有用户
        """
        if user_indices is None:
            # 为所有用户生成嵌入
            base_embeds = self.user_embeds.weight
        else:
            # 为特定用户生成嵌入
            base_embeds = self.user_embeds(user_indices)
        
        # 生成不同意图的用户表示
        interest_embeds = self.interest_gate(base_embeds)
        dependency_embeds = self.dependency_gate(base_embeds)
        
        # 如果有社交网络数据，使用GCN增强
        if self.adj_tensor is not None and user_indices is None:
            # 对所有用户执行GCN
            social_embeds = self.gcn_act(torch.spmm(self.adj_tensor, self.gcn_layer(base_embeds)))
            # 融合不同的表示
            combined_embeds = torch.cat([base_embeds, interest_embeds, dependency_embeds], dim=-1)
        else:
            # 没有社交网络数据或只处理特定用户，只融合意图表示
            combined_embeds = torch.cat([base_embeds, interest_embeds, dependency_embeds], dim=-1)
        
        enhanced_embeds = self.fusion_layer(combined_embeds)
        
        return enhanced_embeds

class EncoderRNN(nn.Module):
    """编码器RNN"""
    
    def __init__(self, user_embed, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed_dim = user_embed.embed_dim
        
        self.user_embed = user_embed
        self.gru = nn.GRU(
            self.embed_dim + 1,  # +1 用于时间间隔特征
            hidden_size,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, input_seq, input_lengths, time_intervals):
        """
        参数:
            input_seq: [batch_size, seq_len]
            input_lengths: [batch_size]
            time_intervals: [batch_size, seq_len]
        """
        # 获取用户嵌入
        embedded = self.user_embed(input_seq)  # [batch_size, seq_len, embed_dim]
        
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
        
        return outputs, hidden

class LongShortTermAttention(nn.Module):
    """长短期注意力，从DisenIDP借鉴"""
    
    def __init__(self, hidden_size):
        super(LongShortTermAttention, self).__init__()
        
        # 长期影响注意力参数
        self.long_query = nn.Linear(hidden_size, hidden_size)
        self.long_key = nn.Linear(hidden_size, hidden_size)
        self.long_value = nn.Linear(hidden_size, hidden_size)
        
        # 短期影响注意力参数
        self.short_query = nn.Linear(hidden_size, hidden_size)
        self.short_key = nn.Linear(hidden_size, hidden_size)
        self.short_value = nn.Linear(hidden_size, hidden_size)
        
        # 融合层
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, encoder_outputs, hidden):
        """
        参数:
            encoder_outputs: [batch_size, seq_len, hidden_size]
            hidden: [n_layers, batch_size, hidden_size]
        """
        # 使用最后一层隐藏状态
        last_hidden = hidden[-1]  # [batch_size, hidden_size]
        
        # 计算长期注意力 (使用第一个节点作为查询)
        first_node = encoder_outputs[:, 0, :]  # [batch_size, hidden_size]
        long_query = self.long_query(first_node).unsqueeze(1)  # [batch_size, 1, hidden_size]
        long_key = self.long_key(encoder_outputs)  # [batch_size, seq_len, hidden_size]
        long_value = self.long_value(encoder_outputs)  # [batch_size, seq_len, hidden_size]
        
        long_scores = torch.bmm(long_query, long_key.transpose(1, 2)) / math.sqrt(encoder_outputs.size(-1))
        long_attn = F.softmax(long_scores, dim=-1)  # [batch_size, 1, seq_len]
        long_context = torch.bmm(long_attn, long_value).squeeze(1)  # [batch_size, hidden_size]
        
        # 计算短期注意力 (使用最后隐藏状态作为查询)
        short_query = self.short_query(last_hidden).unsqueeze(1)  # [batch_size, 1, hidden_size]
        short_key = self.short_key(encoder_outputs)  # [batch_size, seq_len, hidden_size]
        short_value = self.short_value(encoder_outputs)  # [batch_size, seq_len, hidden_size]
        
        short_scores = torch.bmm(short_query, short_key.transpose(1, 2)) / math.sqrt(encoder_outputs.size(-1))
        short_attn = F.softmax(short_scores, dim=-1)  # [batch_size, 1, seq_len]
        short_context = torch.bmm(short_attn, short_value).squeeze(1)  # [batch_size, hidden_size]
        
        # 融合长短期上下文
        context = torch.cat([long_context, short_context], dim=-1)  # [batch_size, hidden_size*2]
        context = self.fusion(context)  # [batch_size, hidden_size]
        
        return context, long_attn.squeeze(1), short_attn.squeeze(1)

class DecoderRNN(nn.Module):
    """解码器RNN"""
    
    def __init__(self, user_embed, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embed_dim = user_embed.embed_dim
        
        self.user_embed = user_embed
        self.dropout = nn.Dropout(dropout)
        self.attention = LongShortTermAttention(hidden_size)
        
        self.gru = nn.GRU(
            self.embed_dim + hidden_size,  # 嵌入 + 上下文向量
            hidden_size,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.out = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        参数:
            input_step: [batch_size]
            last_hidden: [n_layers, batch_size, hidden_size]
            encoder_outputs: [batch_size, seq_len, hidden_size]
        """
        # 获取当前步骤的嵌入
        embedded = self.user_embed(input_step)  # [batch_size, embed_dim]
        embedded = self.dropout(embedded)
        
        # 计算注意力上下文
        context, _, _ = self.attention(encoder_outputs, last_hidden)
        
        # 合并嵌入和上下文
        rnn_input = torch.cat([embedded, context], dim=1)  # [batch_size, embed_dim+hidden_size]
        rnn_input = rnn_input.unsqueeze(1)  # [batch_size, 1, embed_dim+hidden_size]
        
        # GRU计算
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)  # [batch_size, hidden_size]
        
        # 预测下一个用户
        output = torch.cat([output, context], dim=1)  # [batch_size, hidden_size*2]
        output = self.out(output)  # [batch_size, output_size]
        
        return output, hidden

class Seq2SeqModel(nn.Module):
    """序列到序列模型"""
    
    def __init__(self, user_size, embed_dim, hidden_size, n_layers=2, dropout=0.1, adj_tensor=None, pretrained_embeds=None):
        super(Seq2SeqModel, self).__init__()
        
        # 用户嵌入增强
        self.user_embed = UserEmbeddingEnhancement(
            user_size, 
            embed_dim, 
            adj_tensor, 
            pretrained_embeds
        )
        
        # 编码器
        self.encoder = EncoderRNN(
            self.user_embed,
            hidden_size,
            n_layers,
            dropout
        )
        
        # 解码器
        self.decoder = DecoderRNN(
            self.user_embed,
            hidden_size,
            user_size,
            n_layers,
            dropout
        )
        
    def forward(self, src, src_lengths, time_intervals, tgt, teacher_forcing_ratio=0.5):
        """
        参数:
            src: [batch_size, src_len]
            src_lengths: [batch_size]
            time_intervals: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
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
        
        # 解码
        for t in range(1, tgt_len):
            # 解码一个步骤
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t-1, :] = output
            
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # 获取最可能的单词
            _, topi = output.topk(1)
            decoder_input = tgt[:, t] if teacher_force else topi.squeeze(-1)
        
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
        
        # 存储预测
        predictions = []
        
        # 逐步解码
        for _ in range(max_length):
            # 解码一个步骤
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            
            # 获取最可能的用户
            _, topi = output.topk(1)
            decoder_input = topi.squeeze(-1)
            
            # 添加到预测中
            predictions.append(decoder_input.detach())
        
        # 堆叠预测结果
        return torch.stack(predictions, dim=1)  # [batch_size, max_length] 