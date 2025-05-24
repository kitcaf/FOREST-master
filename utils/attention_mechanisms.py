import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GELU(nn.Module):
    """GELU activation function as used in DisenIDP"""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    """Layer normalization module"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class TransformerBlock(nn.Module):
    """Transformer block for attention-based feature extraction"""
    def __init__(self, input_size, d_model=64, n_heads=2, is_layer_norm=True, is_FFN=True, is_future=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()

        assert d_model % n_heads == 0

        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.is_future = is_future
        self.is_FFN = is_FFN

        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_v))
        self.W_o = nn.Parameter(torch.Tensor(self.d_v*n_heads, input_size))

        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.activation = GELU()

        self.is_layer_norm = is_layer_norm
        self.layer_norm = nn.LayerNorm(input_size)

        self.__init_weights__()

    def __init_weights__(self):
        nn.init.normal_(self.W_q, mean=0, std=np.sqrt(2.0 / (self.input_size + self.d_model)))
        nn.init.normal_(self.W_k, mean=0, std=np.sqrt(2.0 / (self.input_size + self.d_model)))
        nn.init.normal_(self.W_v, mean=0, std=np.sqrt(2.0 / (self.input_size + self.d_model)))

        nn.init.xavier_normal_(self.W_o)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        """Feed-forward network"""
        output = self.linear2(self.activation(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        """Scaled dot-product attention"""
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            if self.is_future:
                # Future information
                mask = torch.tril(torch.ones(pad_mask.size()), diagonal=0).bool().to(Q.device)
            else:
                # Historic information
                mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(Q.device)

            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -1e10)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)
        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att

    def multi_head_attention(self, Q, K, V, mask):
        """Multi-head attention mechanism"""
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # For head axis broadcasting.
            mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads* self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))  # (batch_size, max_q_words, input_size)

        return output

    def forward(self, Q, K, V, mask=None):
        """Forward pass"""
        V_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm:
            output = self.layer_norm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            if self.is_FFN:
                output = self.layer_norm(self.FFN(output) + output)
        else:
            output = Q + V_att
            if self.is_FFN:
                output = self.FFN(output) + output
        return output


class LongTermAttention(nn.Module):
    """Long-term attention mechanism from DisenIDP"""
    def __init__(self, input_size, attn_dropout=0.1):
        super(LongTermAttention, self).__init__()

        self.input_size = input_size
        self.W_q = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_k = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_v = nn.Parameter(torch.Tensor(input_size, input_size))

        self.dropout = nn.Dropout(attn_dropout)
        self.layer_norm = nn.LayerNorm(input_size)

        self.__init_weights__()

    def __init_weights__(self):
        nn.init.normal_(self.W_q, mean=0, std=np.sqrt(2.0 / (self.input_size)))
        nn.init.normal_(self.W_k, mean=0, std=np.sqrt(2.0 / (self.input_size)))
        nn.init.normal_(self.W_v, mean=0, std=np.sqrt(2.0 / (self.input_size)))

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        """Scaled dot-product attention for long-term dependencies"""
        _, k_len, _ = K.size()
        temperature = self.input_size ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K = Q_K.repeat(1, k_len, 1)

        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(Q.device)
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -1e10)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)
        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att

    def forward(self, Q, K, V, mask=None):
        """Forward pass"""
        # 将Q从[batch_size, hidden_size]扩展为[batch_size, 1, hidden_size]
        Q = Q.unsqueeze(dim=1)
        Q_ = Q.matmul(self.W_q)
        K_ = K.matmul(self.W_k)
        V_ = V.matmul(self.W_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        
        # 修复：确保输出形状与短期注意力一致 [batch_size, hidden_size]
        # 使用Q作为残差连接，而不是V，因为V的形状是[batch_size, seq_len, hidden_size]
        # 而Q的形状是[batch_size, 1, hidden_size]，更接近我们需要的输出形状
        output = self.layer_norm(Q + V_att)
        output = output.squeeze(1)  # [batch_size, hidden_size]
        
        return output

class ShortTermAttention(nn.Module):
    """Short-term attention mechanism from DisenIDP"""
    def __init__(self, input_size, attn_dropout=0.1):
        super(ShortTermAttention, self).__init__()

        self.input_size = input_size
        self.W_q = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_k = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_v = nn.Parameter(torch.Tensor(input_size, input_size))

        self.dropout = nn.Dropout(attn_dropout)
        self.layer_norm = nn.LayerNorm(input_size)

        self.__init_weights__()

    def __init_weights__(self):
        nn.init.normal_(self.W_q, mean=0, std=np.sqrt(2.0 / (self.input_size)))
        nn.init.normal_(self.W_k, mean=0, std=np.sqrt(2.0 / (self.input_size)))
        nn.init.normal_(self.W_v, mean=0, std=np.sqrt(2.0 / (self.input_size)))

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        """Scaled dot-product attention for short-term dependencies"""
        temperature = self.input_size ** 0.5
        Q_K = torch.einsum("bd,bkd->bk", Q, K) / (temperature + episilon)

        if mask is not None:
            Q_K = Q_K.masked_fill(mask, -1e10)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_k_words)
        Q_K_score = self.dropout(Q_K_score)
        V_att = torch.einsum("bk,bkd->bd", Q_K_score, V)  # (batch_size, input_size)
        return V_att

    def forward(self, Q, K, V, mask=None):
        """Forward pass"""
        Q_ = Q.matmul(self.W_q)
        K_ = K.matmul(self.W_k)
        V_ = V.matmul(self.W_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        output = self.layer_norm(Q + V_att)
        return output