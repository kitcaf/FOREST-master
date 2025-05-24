import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import Constants
import math

from utils.graph_utils import normalize, sparse_mx_to_torch_sparse_tensor
from utils.feature_extraction import DisenIDPFeatureExtractor, IntentAwareSelfGating
from utils.attention_mechanisms import LongTermAttention, ShortTermAttention

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

class UserEmbeddingEnhancement(nn.Module):
    """用户嵌入增强模块，整合DisenIDP特征提取方法"""
    
    def __init__(self, user_size, embed_dim, adj_tensor=None, pretrained_embeds=None, dropout=0.2):
        super(UserEmbeddingEnhancement, self).__init__()
        
        self.user_size = user_size
        self.embed_dim = embed_dim
        
        # 使用DisenIDP的特征提取器
        self.feature_extractor = DisenIDPFeatureExtractor(
            user_size=user_size,
            embed_dim=embed_dim,
            adj_tensor=adj_tensor,
            pretrained_embeds=pretrained_embeds,
            dropout=dropout
        )
        
        # 融合层
        self.fusion_layer = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, user_indices=None):
        """
        参数:
            user_indices: 用户索引。如果为None，则返回所有用户的嵌入
        """
        if user_indices is None:
            # 获取所有用户的嵌入
            return self.feature_extractor.user_embedding.weight
        else:
            # 获取特定用户的嵌入
            return self.feature_extractor.user_embedding(user_indices)

class EncoderRNN(nn.Module):
    """编码器RNN，整合DisenIDP特征提取方法"""
    
    def __init__(self, user_embed, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed_dim = user_embed.embed_dim
        
        self.user_embed = user_embed
        self.feature_extractor = user_embed.feature_extractor
        
        # GRU层处理序列
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

class DecoderRNN(nn.Module):
    """解码器RNN，整合DisenIDP的长短期注意力机制"""
    
    def __init__(self, user_embed, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embed_dim = user_embed.embed_dim
        
        self.user_embed = user_embed
        self.dropout = nn.Dropout(dropout)
        
        # 使用DisenIDP的长短期注意力
        self.long_term_attention = LongTermAttention(hidden_size, attn_dropout=dropout)
        self.short_term_attention = ShortTermAttention(hidden_size, attn_dropout=dropout)
        
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
        
        # 计算长期注意力（使用第一个编码器输出）
        first_encoder_output = encoder_outputs[:, 0, :]  # [batch_size, hidden_size]
        long_term_context = self.long_term_attention(first_encoder_output, encoder_outputs, encoder_outputs)
        
        # 计算短期注意力（使用最后一个隐藏状态）
        last_hidden_state = last_hidden[-1]  # [batch_size, hidden_size]
        short_term_context = self.short_term_attention(last_hidden_state, encoder_outputs, encoder_outputs)
        
        # 修复: 确保长短期上下文形状匹配
        # LongTermAttention 返回 [batch_size, seq_len, hidden_size]
        # ShortTermAttention 返回 [batch_size, hidden_size]
        # 我们需要取长期上下文的最后一个时间步
        if long_term_context.dim() > short_term_context.dim():
            long_term_context = long_term_context[:, -1, :]
        
        # 融合长短期上下文
        context = (long_term_context + short_term_context) / 2  # 简单平均融合
        
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
    """序列到序列模型，整合DisenIDP特征提取方法"""
    
    def __init__(self, user_size, embed_dim, hidden_size, n_layers=2, dropout=0.1, adj_tensor=None, pretrained_embeds=None):
        super(Seq2SeqModel, self).__init__()
        
        # 用户嵌入增强，整合DisenIDP特征
        self.user_embed = UserEmbeddingEnhancement(
            user_size, 
            embed_dim, 
            adj_tensor, 
            pretrained_embeds,
            dropout
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