import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
import Constants
import random

"""
改进的序列到序列模型:
    1. 用户编码器：融合社交图信息
    2. 时序-图联合编码器：LSTM + 动态图特征注入
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

class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.attn_fc = nn.Linear(in_features, out_features)
        self.attn_dropout = nn.Dropout(0.1)
        self.attn_leaky_relu = nn.LeakyReLU(0.2)
        self.attn_softmax = nn.Softmax(dim=1)
        
    def forward(self, x, adj):
        # x: [batch_size, seq_len, in_features]
        # adj: [batch_size, seq_len, seq_len]
        
        attn_output = self.attn_fc(x)  # [batch_size, seq_len, out_features]
        attn_output = self.attn_dropout(attn_output)
        attn_output = self.attn_leaky_relu(attn_output)
        
        # 应用注意力掩码
        attn_output = attn_output * adj
        
        # 归一化注意力
        attn_output = self.attn_softmax(attn_output)
        
        return attn_output

class UserEncoder(nn.Module):
    """User encoder that integrates social graph information"""
    def __init__(self, user_size, embed_dim, dropout=0.1, use_network=False, adj=None):
        super(UserEncoder, self).__init__()
        
        self.user_embedding = nn.Embedding(user_size, embed_dim, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)
        self.use_network = use_network
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        
        # Social network integration
        if use_network and adj is not None:
            self.adj = adj
            self.neighbor_aggregation = nn.Linear(embed_dim, embed_dim)
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: User ID sequence [batch_size, seq_len]
            mask: Mask [batch_size, seq_len]
        Returns:
            User embeddings [batch_size, seq_len, embed_dim]
        """
        # Get user embeddings
        user_embeds = self.user_embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Apply social network integration if enabled
        if self.use_network and hasattr(self, 'adj'):
            batch_size, seq_len = x.size()
            
            # Create enhanced embeddings with neighbor information
            enhanced_user_embeds = torch.zeros_like(user_embeds)
            
            for i in range(batch_size):
                for j in range(seq_len):
                    uid = x[i, j].item()
                    if uid != Constants.PAD:
                        # Get original embedding
                        original = user_embeds[i, j]
                        
                        # Aggregate neighbor embeddings
                        neighbor_embed = self._aggregate_neighbors(uid)
                        
                        if neighbor_embed is not None:
                            # Apply gating mechanism
                            combined = torch.cat([original, neighbor_embed], dim=0)
                            gate_value = self.gate(combined)
                            enhanced_user_embeds[i, j] = gate_value * neighbor_embed + (1 - gate_value) * original
                        else:
                            enhanced_user_embeds[i, j] = original
                    else:
                        enhanced_user_embeds[i, j] = user_embeds[i, j]
            
            # Apply layer normalization
            user_embeds = self.layer_norm(enhanced_user_embeds)
        
        # Apply dropout
        user_embeds = self.dropout(user_embeds)
        
        # Apply mask if provided
        if mask is not None:
            user_embeds = user_embeds * mask.unsqueeze(-1)
        
        return user_embeds
    
    def _aggregate_neighbors(self, user_id):
        """Aggregate embeddings from user's neighbors"""
        if not hasattr(self, 'adj') or user_id >= self.adj.size(0):
            return None
            
        # Get neighbors from adjacency matrix
        neighbors = torch.nonzero(self.adj[user_id]).squeeze(1)
        
        if neighbors.numel() == 0:
            return None
            
        # Get neighbor embeddings
        neighbor_embeds = self.user_embedding(neighbors)
        
        # Mean aggregation
        agg_embed = torch.mean(neighbor_embeds, dim=0)
        
        # Process through linear layer
        return self.neighbor_aggregation(agg_embed)

class TimeEncoder(nn.Module):
    """Time interval encoding module"""
    def __init__(self, embed_dim, dropout=0.1):
        super(TimeEncoder, self).__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, time_intervals):
        """
        Args:
            time_intervals: Time intervals [batch_size, seq_len]
        Returns:
            Time embeddings [batch_size, seq_len, embed_dim]
        """
        # Convert time intervals to embeddings
        time_intervals = time_intervals.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        time_embed = self.time_embedding(time_intervals)  # [batch_size, seq_len, embed_dim]
        
        # Apply layer normalization and dropout
        time_embed = self.layer_norm(time_embed)
        time_embed = self.dropout(time_embed)
        
        return time_embed

class TemporalGraphEncoder(nn.Module):
    """时序-图联合编码器：使用RNN编码时序信息"""
    def __init__(self, embed_dim, hidden_dim, n_layers=1, dropout=0.1, max_seq_length=3000, rnn_type='GRU'):
        super(TemporalGraphEncoder, self).__init__()
        
        # 选择RNN类型
        rnn_class = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        
        # 使用双向RNN
        self.rnn = rnn_class(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,  # 因为是双向的，所以隐藏维度减半
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 使用更合理的初始化
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask, time_intervals=None):
        """
        参数:
            x: 用户嵌入 [batch_size, seq_len, embed_dim]
            mask: 掩码 [batch_size, seq_len]
            time_intervals: 时间间隔 [batch_size, seq_len]
        返回:
            encoded: 编码后的序列 [batch_size, seq_len, hidden_dim]
        """
        # 获取有效长度
        lengths = mask.sum(dim=1).long()
        lengths = torch.clamp(lengths, min=1)
        
        # 应用dropout
        x = self.dropout(x)
        
        # 打包序列
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 应用RNN
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
        
        # 应用层归一化和dropout
        encoded = self.layer_norm(output)
        encoded = self.dropout(encoded)
        
        return encoded

class EfficientDecoder(nn.Module):
    """高效解码器：基于联合特征预测未来用户"""
    def __init__(self, hidden_dim, user_size, dropout=0.1):
        super(EfficientDecoder, self).__init__()
        
        # 使用更合理的初始化
        self.output_layer = nn.Linear(hidden_dim, user_size)
        
        # 使用Xavier初始化，但缩小范围
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 添加位置编码
        self.position_embedding = nn.Embedding(4, hidden_dim)  # 最多4个位置
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)
        
        # 增加dropout
        self.dropout = nn.Dropout(dropout)
        
        # 添加L2正则化
        self.l2_reg = 1e-5
        
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
        
        # 应用层归一化
        last_hidden = self.layer_norm(last_hidden)
        
        # 创建输出张量
        outputs = torch.zeros(batch_size, 3, user_size, device=encoder_output.device)
        
        # 对每个时间步分别生成输出
        for t in range(3):
            # 获取位置编码
            pos_embed = self.position_embedding(torch.tensor(t+1, device=encoder_output.device))
            pos_embed = pos_embed.expand(batch_size, -1)  # [batch_size, hidden_dim]
            
            # 将位置编码添加到隐藏状态
            current_hidden = last_hidden + pos_embed * 0.1
            
            # 应用dropout
            current_hidden = self.dropout(current_hidden)
            
            # 应用输出层
            logits = self.output_layer(current_hidden)
            
            # 应用L2正则化
            l2_loss = self.l2_reg * torch.norm(self.output_layer.weight, p=2)
            if self.training:
                # 在训练模式下添加正则化损失
                logits = logits - l2_loss.expand_as(logits) * 0.01
            
            outputs[:, t] = logits
        
        return outputs

class SocialSeq2SeqModel(nn.Module):
    """Sequence-to-sequence model with social graph integration"""
    def __init__(self, user_size, embed_dim, hidden_dim, n_layers=2, dropout=0.1, 
                 use_network=False, adj=None, teacher_forcing_ratio=0.5):
        super(SocialSeq2SeqModel, self).__init__()
        
        # User encoder
        self.user_encoder = UserEncoder(
            user_size=user_size,
            embed_dim=embed_dim,
            dropout=dropout,
            use_network=use_network,
            adj=adj
        )
        
        # Time encoder
        self.time_encoder = TimeEncoder(
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=embed_dim * 2,  # User embedding + time embedding
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim * 2,  # Match bidirectional encoder output
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # Context + hidden state
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, user_size)
        )
        
        # Initialize parameters
        self._init_parameters()
        
        # Store attributes
        self.user_size = user_size
        self.hidden_dim = hidden_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                if len(param.shape) >= 2:  # Only apply Xavier init to matrices
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:  # For vectors (like bias terms)
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def create_mask(self, src):
        """Create source sequence mask"""
        return (src != Constants.PAD).float()  # [batch_size, src_len]
    
    def forward(self, src, src_lengths, tgt=None, time_intervals=None):
        """
        Forward pass
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
            tgt: Target sequence [batch_size, tgt_len]
            time_intervals: Time intervals [batch_size, src_len]
            
        Returns:
            outputs: Decoder outputs [batch_size, tgt_len-1, user_size]
        """
        batch_size = src.size(0)
        
        # Create mask
        src_mask = self.create_mask(src)
        
        # Encode users
        user_embedded = self.user_encoder(src, src_mask)  # [batch_size, src_len, embed_dim]
        
        # Encode time intervals
        if time_intervals is not None:
            time_embedded = self.time_encoder(time_intervals)  # [batch_size, src_len, embed_dim]
        else:
            # Create default time embeddings if not provided
            time_embedded = torch.zeros_like(user_embedded)
        
        # Concatenate user and time embeddings
        encoder_input = torch.cat([user_embedded, time_embedded], dim=2)  # [batch_size, src_len, embed_dim*2]
        
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            encoder_input, 
            src_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Encoder forward pass
        packed_outputs, (hidden, cell) = self.encoder_lstm(packed_input)
        
        # Unpack sequence
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # Process bidirectional encoder hidden states
        hidden = self._reshape_bidirectional_states(hidden, batch_size)
        cell = self._reshape_bidirectional_states(cell, batch_size)
        
        # Determine target sequence length
        tgt_len = 3  # Default: predict 3 future users
        if tgt is not None:
            tgt_len = tgt.size(1) - 1  # Exclude BOS token
        
        # Initialize decoder input with BOS token
        decoder_input = torch.full((batch_size, 1), Constants.BOS, device=src.device)
        
        # Create output tensor
        outputs = torch.zeros(batch_size, tgt_len, self.user_size, device=src.device)
        
        # Decoder forward pass
        for t in range(tgt_len):
            # Get current input embedding
            current_input = self.user_encoder.user_embedding(decoder_input)  # [batch_size, 1, embed_dim]
            
            # Decoder step
            decoder_output, (hidden, cell) = self.decoder_lstm(current_input, (hidden, cell))
            
            # Attention mechanism
            query = decoder_output.transpose(0, 1)  # [1, batch_size, hidden_dim*2]
            key = encoder_outputs.transpose(0, 1)   # [src_len, batch_size, hidden_dim*2]
            value = encoder_outputs.transpose(0, 1) # [src_len, batch_size, hidden_dim*2]
            
            # Apply attention
            attn_output, _ = self.attention(query, key, value, key_padding_mask=(src_mask == 0))
            attn_output = attn_output.transpose(0, 1)  # [batch_size, 1, hidden_dim*2]
            
            # Concatenate decoder output and attention context
            combined = torch.cat([decoder_output, attn_output], dim=2)  # [batch_size, 1, hidden_dim*4]
            
            # Generate output
            output = self.output_projection(combined).squeeze(1)  # [batch_size, user_size]
            outputs[:, t] = output
            
            # Next input - teacher forcing or use prediction
            if self.training and tgt is not None and random.random() < self.teacher_forcing_ratio:
                # Teacher forcing - use real target
                decoder_input = tgt[:, t+1].unsqueeze(1)  # [batch_size, 1]
            else:
                # Use model prediction
                top1 = output.argmax(1).unsqueeze(1)  # [batch_size, 1]
                decoder_input = top1
        
        return outputs
    
    def _reshape_bidirectional_states(self, state, batch_size):
        """Reshape bidirectional LSTM states for the decoder"""
        num_layers = state.size(0) // 2
        hidden_dim = state.size(2)
        
        # Reshape from [num_layers*2, batch_size, hidden_dim] to [num_layers, batch_size, hidden_dim*2]
        # by concatenating forward and backward states
        reshaped = torch.zeros(num_layers, batch_size, hidden_dim*2, device=state.device)
        
        for i in range(num_layers):
            # Concatenate forward and backward states
            forward_state = state[i*2]
            backward_state = state[i*2+1]
            reshaped[i] = torch.cat([forward_state, backward_state], dim=1)
        
        return reshaped
    
    def generate(self, src, src_lengths, max_len=3, time_intervals=None):
        """
        Generate sequence without teacher forcing
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
            max_len: Maximum generation length
            time_intervals: Time intervals [batch_size, src_len]
            
        Returns:
            generated_seq: Generated sequence [batch_size, max_len]
            generated_probs: Generated probabilities [batch_size, max_len, user_size]
        """
        batch_size = src.size(0)
        
        # Forward pass to get predictions
        with torch.no_grad():
            outputs = self.forward(src, src_lengths, time_intervals=time_intervals)
        
        # Extract prediction logits
        logits = outputs  # [batch_size, max_len, user_size]
        
        # Store generated sequence and probabilities
        generated_seq = torch.zeros(batch_size, max_len, dtype=torch.long, device=src.device)
        generated_probs = torch.zeros(batch_size, max_len, self.user_size, device=src.device)
        
        # Track generated users to avoid repetition
        generated_users = [set() for _ in range(batch_size)]
        
        # Generate for each position
        for t in range(max_len):
            # Get current position output
            step_output = logits[:, t].clone()  # [batch_size, user_size]
            
            # Apply mask to prevent generating already seen users
            for i in range(batch_size):
                # Mask users from source sequence
                for j in range(src.size(1)):
                    user_id = src[i, j].item()
                    if user_id != Constants.PAD:
                        step_output[i, user_id] = float('-inf')
                
                # Mask already generated users
                for user_id in generated_users[i]:
                    step_output[i, user_id] = float('-inf')
            
            # Apply softmax with temperature
            temperature = 0.7
            step_probs = F.softmax(step_output / temperature, dim=1)
            
            # Handle NaN values
            if torch.isnan(step_probs).any():
                step_probs = torch.nan_to_num(step_probs, nan=0.0)
                # Renormalize
                step_probs = step_probs / (step_probs.sum(dim=1, keepdim=True) + 1e-10)
            
            # Store probabilities
            generated_probs[:, t] = step_probs
            
            # Sample next user
            top1 = step_output.argmax(1)  # [batch_size]
            
            # Store generated user
            generated_seq[:, t] = top1
            
            # Update generated users set
            for i in range(batch_size):
                generated_users[i].add(top1[i].item())
        
        return generated_seq, generated_probs 