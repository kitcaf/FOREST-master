import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
import Constants

"""
序列到序列模型 (seq2seq_model.py):
    编码器-解码器架构
    社交网络增强
    位置编码
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
    def __init__(self, d_model, max_seq_length=1000):
        super(PositionalEncoding, self).__init__()
        
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
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    """编码器模块"""
    def __init__(self, user_size, embed_dim, hidden_dim, n_layers=2, dropout=0.1, use_network=False, adj=None, net_dict=None):
        super(Encoder, self).__init__()
        
        self.user_embedding = nn.Embedding(user_size, embed_dim, padding_idx=Constants.PAD)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.use_network = use_network
        
        if use_network:
            self.adj = adj
            self.net_dict = net_dict
            self.gcn1 = nn.Linear(embed_dim, embed_dim)
            self.gcn2 = nn.Linear(embed_dim, embed_dim)
            self.nnl1 = 25  # 一阶邻居采样数
            self.nnl2 = 10  # 二阶邻居采样数
            
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def neighbor_sampling(self, nodes, sample_size):
        """采样邻居节点"""
        result = []
        for node in nodes:
            # 获取邻居
            if node in self.net_dict and len(self.net_dict[node]) > 0:
                neighbors = list(self.net_dict[node])
                # 如果邻居数量不足，则重复采样
                if len(neighbors) < sample_size:
                    neighbors = neighbors * (sample_size // len(neighbors) + 1)
                # 随机采样
                sampled = np.random.choice(neighbors, sample_size, replace=False)
                result.append(sampled)
            else:
                # 如果没有邻居，则采样随机节点
                result.append(np.random.randint(0, len(self.net_dict), sample_size))
        return np.array(result)
        
    def forward(self, src, src_lengths):
        # src: [batch_size, src_len]
        batch_size, src_len = src.shape
        
        # 基本嵌入
        embedded = self.user_embedding(src)  # [batch_size, src_len, embed_dim]
        embedded = self.pos_encoder(embedded)
        
        # 社交网络增强
        if self.use_network:
            # 采样一阶和二阶邻居
            nb1 = self.neighbor_sampling(src.view(-1).cpu().numpy(), self.nnl1)  # [batch*src_len, nnl1]
            nb2 = self.neighbor_sampling(nb1.reshape(-1), self.nnl2)  # [batch*src_len*nnl1, nnl2]
            
            # 获取邻居嵌入
            nb2_embed = self.user_embedding(torch.LongTensor(nb2).to(src.device))  # [batch*src_len*nnl1, nnl2, embed_dim]
            
            # 图卷积聚合
            nf2 = F.relu(self.gcn2(nb2_embed).mean(dim=1))  # [batch*src_len*nnl1, embed_dim]
            nf2 = nf2.view(-1, self.nnl1, embedded.size(2))  # [batch*src_len, nnl1, embed_dim]
            
            nf1 = F.relu(self.gcn1(nf2).mean(dim=1))  # [batch*src_len, embed_dim]
            nf1 = nf1.view(batch_size, src_len, -1)  # [batch_size, src_len, embed_dim]
            
            # 结合原始嵌入和图增强嵌入
            embedded = embedded + nf1
        
        embedded = self.dropout(embedded)
        
        # 打包序列以处理变长序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 通过LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        
        # 解包序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # 双向LSTM的输出合并
        # hidden: [n_layers*2, batch_size, hidden_dim]
        # 取最后一层的隐藏状态
        hidden_fwd = hidden[-2, :, :]  # 前向最后一层
        hidden_bwd = hidden[-1, :, :]  # 后向最后一层
        hidden = torch.cat((hidden_fwd, hidden_bwd), dim=1)  # [batch_size, hidden_dim*2]
        
        return outputs, hidden  # [batch_size, src_len, hidden_dim*2], [batch_size, hidden_dim*2]

class Decoder(nn.Module):
    """解码器模块 - 简化版，没有注意力机制"""
    def __init__(self, user_size, embed_dim, hidden_dim, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.user_embedding = nn.Embedding(user_size, embed_dim, padding_idx=Constants.PAD)
        self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=False)
        self.fc_out = nn.Linear(hidden_dim, user_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hidden_dim]
        
        # 嵌入输入
        embedded = self.dropout(self.user_embedding(input))  # [batch_size, embed_dim]
        
        # 添加序列长度维度
        embedded = embedded.unsqueeze(0)  # [1, batch_size, embed_dim]
        
        # 通过RNN
        output, hidden = self.rnn(embedded, hidden)  # [1, batch_size, hidden_dim], [n_layers, batch_size, hidden_dim]
        
        # 移除序列长度维度
        output = output.squeeze(0)  # [batch_size, hidden_dim]
        
        # 预测下一个用户
        prediction = self.fc_out(output)  # [batch_size, user_size]
        
        return prediction, hidden

class Seq2SeqModel(nn.Module):
    """序列到序列模型 - 简化版，没有注意力机制"""
    def __init__(self, user_size, embed_dim, hidden_dim, n_layers=2, dropout=0.1, 
                 use_network=False, adj=None, net_dict=None, teacher_forcing_ratio=0.5):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = Encoder(user_size, embed_dim, hidden_dim, n_layers, dropout, use_network, adj, net_dict)
        self.decoder = Decoder(user_size, embed_dim, hidden_dim, n_layers, dropout)
        
        self.user_size = user_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 添加一个投影层，将编码器的隐藏状态投影到解码器的维度
        self.hidden_proj = nn.Linear(hidden_dim*2, hidden_dim)
        
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=None):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio
            
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # 编码
        _, hidden = self.encoder(src, src_lengths)  # [batch_size, hidden_dim*2]
        
        # 投影隐藏状态
        hidden = self.hidden_proj(hidden)  # [batch_size, hidden_dim]
        
        # 调整隐藏状态的形状以适应解码器
        hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)  # [n_layers, batch_size, hidden_dim]
        
        # 初始化解码器输入
        decoder_input = tgt[:, 0]  # 第一个输入是目标序列的第一个元素
        
        # 存储预测结果
        outputs = torch.zeros(batch_size, tgt_len, self.user_size).to(src.device)
        
        # 逐步解码
        for t in range(1, tgt_len):
            # 解码一步
            output, hidden = self.decoder(decoder_input, hidden)
            
            # 存储当前步的预测
            outputs[:, t, :] = output
            
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # 获取下一步的输入
            top1 = output.argmax(1)
            decoder_input = tgt[:, t] if teacher_force else top1
            
        return outputs
    
    def generate(self, src, src_lengths, max_len=100, start_tokens=None):
        """生成序列"""
        batch_size = src.size(0)
        
        # 编码
        _, hidden = self.encoder(src, src_lengths)
        
        # 投影隐藏状态
        hidden = self.hidden_proj(hidden)
        
        # 调整隐藏状态的形状以适应解码器
        hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        # 初始化解码器输入
        if start_tokens is None:
            decoder_input = torch.tensor([Constants.BOS] * batch_size).to(src.device)
        else:
            decoder_input = start_tokens
        
        # 存储生成的序列
        generated_seq = torch.zeros(batch_size, max_len, dtype=torch.long).to(src.device)
        generated_probs = torch.zeros(batch_size, max_len, self.user_size).to(src.device)
        
        # 已生成的用户集合（避免重复）
        generated_users = [set() for _ in range(batch_size)]
        
        # 逐步生成
        for t in range(max_len):
            # 解码一步
            output, hidden = self.decoder(decoder_input, hidden)
            
            # 应用掩码防止生成已经出现过的用户
            for i in range(batch_size):
                if decoder_input[i].item() not in [Constants.PAD, Constants.EOS]:
                    generated_users[i].add(decoder_input[i].item())
                for user_id in generated_users[i]:
                    output[i, user_id] = float('-inf')
            
            # 存储概率分布
            generated_probs[:, t, :] = F.softmax(output, dim=1)
            
            # 采样下一个用户
            if t < 5:  # 前几步使用贪婪搜索
                top1 = output.argmax(1)
            else:  # 后面使用采样
                top1 = torch.multinomial(F.softmax(output, dim=1), 1).squeeze()
            
            # 存储生成的用户
            generated_seq[:, t] = top1
            
            # 更新解码器输入
            decoder_input = top1
            
            # 如果生成了EOS，则停止
            if (top1 == Constants.EOS).all():
                break
                
        return generated_seq, generated_probs 