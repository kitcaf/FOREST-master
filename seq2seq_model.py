"""
Graph-Augmented Seq2Seq模型（自回归）
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
        
        # 构建邻接字典，用于快速查找邻居
        self.adj_dict = {}
        if adj_matrix is not None and hasattr(adj_matrix, '_indices'):
            try:
                # 从稀疏张量构建邻接字典
                indices = adj_matrix._indices().cpu().numpy()
                for i in range(indices.shape[1]):
                    src, dst = indices[0, i], indices[1, i]
                    if src not in self.adj_dict:
                        self.adj_dict[src] = []
                    self.adj_dict[src].append(dst)
            except Exception as e:
                print(f"解码器构建邻接字典时出错: {str(e)}")
        
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
        
        # 如果提供了上一个节点且有邻接字典，应用社交网络约束
        if last_node is not None and hasattr(self, 'adj_dict') and self.adj_dict:
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
                    try:
                        # 获取当前节点的邻居
                        if node_idx in self.adj_dict and self.adj_dict[node_idx]:
                            neighbors = self.adj_dict[node_idx]
                            for neighbor in neighbors:
                                if neighbor < self.output_size:
                                    mask[i, neighbor] = 1.0  # 允许邻居节点
                    except Exception as e:
                        print(f"警告: 获取节点 {node_idx} 的邻居时出错: {e}")
                        # 出错时不应用约束
                        mask[i, :] = 1.0
                    
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
        
        # 确保用户词汇表大小合理
        if user_size > 1000000:
            print(f"警告: 用户词汇表大小 ({user_size}) 过大，可能导致数值问题")
        
        # 保存用户大小
        self.user_size = user_size
        
        # 检查是否有社交图信息
        self.has_social_info = adj_tensor is not None
        
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
        
        # 保存邻接矩阵，用于社交网络修正
        self.adj_tensor = adj_tensor
        
        # 初始化邻接字典
        self.adj_dict = {}
        if hasattr(adj_tensor, '_indices') and adj_tensor is not None:
            try:
                # 从稀疏张量构建邻接字典
                indices = adj_tensor._indices().cpu().numpy()
                for i in range(indices.shape[1]):
                    src, dst = indices[0, i], indices[1, i]
                    if src not in self.adj_dict:
                        self.adj_dict[src] = []
                    self.adj_dict[src].append(dst)
                print(f"从稀疏张量构建邻接字典成功，共有 {len(self.adj_dict)} 个节点有连接")
            except Exception as e:
                print(f"构建邻接字典时出错: {str(e)}")
        
        # 使用Xavier初始化所有参数，增加数值稳定性
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化模型权重，增加数值稳定性"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Xavier初始化
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GRU):
            # 对GRU参数进行初始化
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def has_social_graph(self):
        """检查模型是否有社交图信息"""
        return hasattr(self, 'adj_dict') and self.adj_dict and len(self.adj_dict) > 0
    
    def apply_social_correction(self, output_probs, last_node, alpha=None):
        """
        应用社交网络修正到输出概率
        
        参数:
            output_probs: [batch_size, vocab_size] - 原始输出概率
            last_node: [batch_size] - 上一个预测的节点
            alpha: 社交网络权重 (如果为None，则使用Constants.SOCIAL_ALPHA)
            
        返回:
            corrected_probs: [batch_size, vocab_size] - 修正后的概率
        """
        if alpha is None:
            alpha = Constants.SOCIAL_ALPHA
            
        # 检查是否有社交图信息
        if not self.has_social_graph():
            return output_probs
            
        # 检查last_node是否为None
        if last_node is None:
            return output_probs
            
        batch_size, vocab_size = output_probs.shape
        device = output_probs.device
        
        # 创建社交网络概率分布
        social_probs = torch.zeros_like(output_probs)
        
        # 对每个batch单独处理
        for i in range(batch_size):
            node_idx = last_node[i].item()
            
            # 如果是特殊标记，不应用约束
            if node_idx < 4:  # 小于4的是特殊标记
                social_probs[i] = 1.0 / vocab_size  # 均匀分布
            else:
                # 获取当前节点的邻居
                try:
                    if node_idx in self.decoder.adj_dict and self.decoder.adj_dict[node_idx]:
                        neighbors = self.decoder.adj_dict[node_idx]
                        # 为邻居节点分配概率
                        for neighbor in neighbors:
                            if neighbor < vocab_size:
                                social_probs[i, neighbor] = 1.0
                        
                        # 归一化邻居概率
                        if social_probs[i].sum() > 0:
                            social_probs[i] = social_probs[i] / social_probs[i].sum()
                        else:
                            social_probs[i] = 1.0 / vocab_size  # 如果没有有效邻居，使用均匀分布
                    else:
                        # 如果没有邻居，使用均匀分布
                        social_probs[i] = 1.0 / vocab_size
                except Exception as e:
                    print(f"警告: 社交修正时出错: {e}")
                    # 出错时使用均匀分布
                    social_probs[i] = 1.0 / vocab_size
                
                # 特殊标记总是可用
                special_token_prob = 0.01
                social_probs[i, :4] = special_token_prob
                # 重新归一化
                if social_probs[i].sum() > 0:
                    social_probs[i] = social_probs[i] / social_probs[i].sum()
        
        # 组合原始概率和社交网络概率
        corrected_probs = (1 - alpha) * output_probs + alpha * social_probs
        
        return corrected_probs
    
    def calculate_social_consistency_loss(self, pred_node, prev_node):
        """
        计算社交一致性损失
        
        参数:
            pred_node: [batch_size] - 预测的节点
            prev_node: [batch_size] - 上一个节点
            
        返回:
            loss: 社交一致性损失
        """
        if not self.has_social_graph():
            return torch.tensor(0.0, device=pred_node.device)
            
        batch_size = pred_node.size(0)
        device = pred_node.device
        batch_loss = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            if prev_node[i].item() < 4 or pred_node[i].item() < 4:
                # 特殊标记不计算社交一致性损失
                continue
                
            try:
                # 获取两个节点之间的边权重
                if isinstance(self.adj_tensor, torch.Tensor) and self.adj_tensor.is_sparse:
                    indices = self.adj_tensor._indices()
                    values = self.adj_tensor._values()
                    
                    # 查找从prev_node到pred_node的边
                    mask = (indices[0] == prev_node[i].item()) & (indices[1] == pred_node[i].item())
                    
                    if mask.any():
                        # 找到边，使用边权重
                        edge_weight = values[mask][0]
                        batch_loss[i] = -torch.log(edge_weight + 1e-10)
                    else:
                        # 没有边，使用较大的损失
                        batch_loss[i] = 10.0
                else:
                    # 对于非稀疏矩阵
                    edge_weight = self.adj_tensor[prev_node[i].item(), pred_node[i].item()]
                    if edge_weight > 0:
                        batch_loss[i] = -torch.log(edge_weight + 1e-10)
                    else:
                        batch_loss[i] = 10.0
            except Exception as e:
                print(f"警告: 计算社交一致性损失时出错: {e}")
                batch_loss[i] = 0.0
        
        return batch_loss.mean()
    
    def _beam_search(self, encoder_outputs, hidden, max_length, beam_size, use_social_correction=False, temperature=1.0):
        """
        使用束搜索进行解码
        参数:
            encoder_outputs: 编码器输出 [batch_size, src_len, hidden_size]
            hidden: 编码器最终隐藏状态
            max_length: 最大生成长度
            beam_size: 束大小
            use_social_correction: 是否使用社交网络修正
            temperature: 温度参数，控制输出分布的平滑度
        返回:
            decoded_words: [batch_size, max_length]
            attentions: 注意力权重列表
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # 为每个批次单独进行束搜索
        final_predictions = torch.zeros(batch_size, max_length, device=device).long()
        final_predictions[:, 0] = Constants.BOS  # 第一个token是BOS
        all_attentions = []
        
        # 检查是否有社交图信息
        has_social_graph = self.has_social_graph()
        if use_social_correction and not has_social_graph:
            print("警告: 请求使用社交网络修正，但没有社交图信息。将不使用社交修正。")
            use_social_correction = False
        
        for batch_idx in range(batch_size):
            # 获取当前批次的编码器输出和隐藏状态
            batch_encoder_outputs = encoder_outputs[batch_idx:batch_idx+1].repeat(beam_size, 1, 1)
            
            if isinstance(hidden, tuple):  # LSTM
                batch_hidden = (hidden[0][:, batch_idx:batch_idx+1].repeat(1, beam_size, 1),
                               hidden[1][:, batch_idx:batch_idx+1].repeat(1, beam_size, 1))
            else:  # GRU
                batch_hidden = hidden[:, batch_idx:batch_idx+1].repeat(1, beam_size, 1)
            
            # 初始化束
            beams = [(torch.tensor([Constants.BOS], device=device), 0.0, batch_hidden, None, [])]  # (序列, 分数, 隐藏状态, 上一个节点, 注意力)
            
            # 束搜索
            for t in range(1, max_length):
                candidates = []
                
                for seq, score, h, last_node, attns in beams:
                    if seq[-1].item() == Constants.EOS:
                        # 如果序列已经结束，则保持原样
                        candidates.append((seq, score, h, last_node, attns))
                        continue
                    
                    # 准备输入
                    decoder_input = seq[-1].view(1)
                    
                    # 解码一步
                    output, new_h, attn_weights = self.decoder(decoder_input, h, batch_encoder_outputs, last_node)
                    
                    # 应用温度
                    if temperature != 1.0:
                        output = output / temperature
                    
                    # 应用社交网络修正
                    if use_social_correction and last_node is not None:
                        # 将logits转换为概率
                        output_probs = torch.softmax(output, dim=-1)
                        
                        # 应用社交修正
                        corrected_probs = self.apply_social_correction(output_probs, last_node)
                        
                        # 转回logits域
                        output = torch.log(corrected_probs + Constants.EPS)
                    else:
                        # 如果不使用社交修正，直接转换为概率再转回logits
                        output = torch.log_softmax(output, dim=-1)
                    
                    # 获取前beam_size个最可能的单词
                    log_probs, indices = output.topk(beam_size)
                    
                    for i in range(beam_size):
                        word_idx = indices[0, i].item()
                        word_score = log_probs[0, i].item()
                        
                        # 计算新序列的分数
                        new_score = score + word_score
                        
                        # 创建新序列
                        new_seq = torch.cat([seq, torch.tensor([word_idx], device=device)])
                        
                        # 存储注意力权重
                        new_attns = attns + [attn_weights] if attn_weights is not None else attns
                        
                        # 添加到候选列表
                        candidates.append((new_seq, new_score, new_h, torch.tensor([word_idx], device=device), new_attns))
                
                # 按分数排序并选择前beam_size个
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
            
            # 选择最高分的序列
            best_seq, _, _, _, best_attns = beams[0]
            
            # 填充到最大长度
            padded_seq = torch.full((max_length,), Constants.PAD, device=device)
            padded_seq[:len(best_seq)] = best_seq
            
            # 存储结果
            final_predictions[batch_idx] = padded_seq
            all_attentions.extend(best_attns)
        
        return final_predictions, all_attentions
    
    def _greedy_decode(self, encoder_outputs, encoder_hidden, max_length, use_social_correction=False, temperature=1.0):
        """
        贪婪解码 - 每一步选择概率最高的节点
        
        参数:
            encoder_outputs: [batch_size, seq_len, hidden_size] - 编码器输出
            encoder_hidden: [n_layers, batch_size, hidden_size] - 编码器最终隐藏状态
            max_length: 最大解码长度
            use_social_correction: 是否使用社交网络修正
            temperature: 温度参数，控制输出分布的平滑度
            
        返回:
            decoded_ids: [batch_size, max_length] - 解码的序列
            decoder_outputs: [batch_size, max_length, vocab_size] - 解码器在每一步的输出概率
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # 检查是否有社交图信息
        if use_social_correction and not self.has_social_graph():
            print("警告: 请求了社交修正，但没有社交图信息。将禁用社交修正。")
            use_social_correction = False
            
        # 初始化解码器输入 (BOS标记)
        decoder_input = torch.tensor([Constants.BOS] * batch_size, device=device)
        
        # 初始化解码器隐藏状态
        decoder_hidden = encoder_hidden
        
        # 存储解码结果
        decoded_ids = []
        decoder_outputs = []
        
        # 存储上一个预测的节点 (用于社交修正)
        prev_node = None
        
        # 逐步解码
        for t in range(max_length):
            # 解码器前向传播
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, prev_node
            )
            
            # 应用温度
            if temperature != 1.0:
                decoder_output = decoder_output / temperature
                
            # 应用社交网络修正 (如果启用)
            if use_social_correction and t > 0:  # 第一步没有前一个节点，跳过
                decoder_output = self.apply_social_correction(
                    F.softmax(decoder_output, dim=-1), prev_node
                )
            else:
                decoder_output = F.softmax(decoder_output, dim=-1)
                
            # 贪婪选择
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()  # 下一步的输入
            
            # 保存结果
            decoded_ids.append(decoder_input)
            decoder_outputs.append(decoder_output)
            
            # 更新前一个节点
            prev_node = decoder_input
            
            # 如果所有序列都到达EOS，提前停止
            if (decoder_input == Constants.EOS).all():
                break
                
        # 堆叠结果
        decoded_ids = torch.stack(decoded_ids, dim=1)  # [batch_size, seq_len]
        decoder_outputs = torch.stack(decoder_outputs, dim=1)  # [batch_size, seq_len, vocab_size]
        
        return decoded_ids, decoder_outputs

    def _teacher_forcing(self, encoder_outputs, encoder_hidden, tgt_seq, teacher_forcing_ratio=0.5, noise_ratio=0.0):
        """
        使用教师强制进行训练
        
        参数:
            encoder_outputs: [batch_size, src_len, hidden_size] - 编码器输出
            encoder_hidden: [n_layers, batch_size, hidden_size] - 编码器最终隐藏状态
            tgt_seq: [batch_size, tgt_len] - 目标序列
            teacher_forcing_ratio: 使用教师强制的概率
            noise_ratio: 噪声注入率，用于增强鲁棒性
            
        返回:
            outputs: [batch_size, tgt_len-1, vocab_size] - 每一步的输出概率
            pred_nodes: [batch_size, tgt_len-1] - 预测的节点序列
            last_node: [batch_size] - 最后一个预测的节点
        """
        batch_size = encoder_outputs.size(0)
        tgt_len = tgt_seq.size(1)
        tgt_vocab_size = self.decoder.output_size
        device = encoder_outputs.device
        
        # 输出张量初始化
        outputs = torch.zeros(batch_size, tgt_len-1, tgt_vocab_size).to(device)
        
        # 存储用于计算社交一致性损失的预测节点
        pred_nodes = []
        
        # 第一个解码输入是BOS标记
        decoder_input = tgt_seq[:, 0]  # [batch_size]
        decoder_hidden = encoder_hidden
        
        # 用于社交约束的上一个节点
        last_node = None
        
        # 解码
        for t in range(1, tgt_len):
            # 解码一个步骤
            output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_node
            )
            
            outputs[:, t-1, :] = output
            
            # 获取模型预测
            _, topi = output.topk(1)
            predicted = topi.squeeze(-1)
            
            # 存储预测节点
            pred_nodes.append(predicted)
            
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # 如果使用教师强制，使用真实标签；否则使用模型预测
            if teacher_force:
                decoder_input = tgt_seq[:, t]
            else:
                decoder_input = predicted
                
                # 噪声注入 - 随机替换一些节点以增强鲁棒性
                if noise_ratio > 0:
                    noise_mask = torch.rand(batch_size, device=device) < noise_ratio
                    if noise_mask.any():
                        # 生成随机节点
                        random_nodes = torch.randint(4, tgt_vocab_size, (noise_mask.sum(),), device=device)
                        decoder_input[noise_mask] = random_nodes
            
            # 更新上一个节点（用于社交约束）
            last_node = decoder_input
        
        # 将预测节点列表转换为张量
        if pred_nodes:
            pred_nodes = torch.stack(pred_nodes, dim=1)  # [batch_size, tgt_len-1]
        else:
            pred_nodes = torch.zeros(batch_size, 0, device=device).long()
        
        return outputs, pred_nodes, last_node

    def forward(self, src_seq, tgt_seq=None, use_beam_search=False, beam_size=5, use_social_correction=False, temperature=1.0, teacher_forcing_ratio=0.5, noise_ratio=0.0):
        """
        模型前向传播
        
        参数:
            src_seq: [batch_size, src_len] - 源序列
            tgt_seq: [batch_size, tgt_len] - 目标序列 (训练时提供)
            use_beam_search: 是否使用束搜索解码
            beam_size: 束搜索大小
            use_social_correction: 是否使用社交网络修正
            temperature: 温度参数，控制输出分布的平滑度
            teacher_forcing_ratio: 使用教师强制的概率
            noise_ratio: 噪声注入率，用于增强鲁棒性
            
        返回:
            训练模式:
                outputs: [batch_size, tgt_len, vocab_size] - 每一步的输出概率
                pred_nodes: [batch_size, tgt_len] - 预测的节点序列
                last_node: [batch_size] - 最后一个预测的节点
            
            推理模式:
                pred_nodes: [batch_size, max_len] - 预测的节点序列
                outputs: [batch_size, max_len, vocab_size] - 每一步的输出概率
        """
        batch_size = src_seq.size(0)
        seq_len = src_seq.size(1)
        device = src_seq.device
        
        # 计算序列长度 - 非填充部分的长度
        src_lengths = torch.sum(src_seq != Constants.PAD, dim=1).long()
        
        # 如果序列长度为0，设为1，避免pack_padded_sequence出错
        src_lengths = torch.clamp(src_lengths, min=1)
        
        # 创建时间间隔 - 如果没有提供，使用全1张量
        time_intervals = torch.ones(batch_size, seq_len).to(device)
        
        # 编码源序列
        encoder_outputs, encoder_hidden = self.encoder(src_seq, src_lengths, time_intervals)
        
        # 训练模式
        if tgt_seq is not None:
            # 使用教师强制训练
            return self._teacher_forcing(encoder_outputs, encoder_hidden, tgt_seq, teacher_forcing_ratio, noise_ratio)
        
        # 推理模式
        max_length = Constants.MAX_SEQ_LEN  # 最大解码长度
        
        # 束搜索解码
        if use_beam_search:
            return self._beam_search(
                encoder_outputs, encoder_hidden, max_length, 
                beam_size, use_social_correction, temperature
            )
        
        # 贪婪解码
        return self._greedy_decode(
            encoder_outputs, encoder_hidden, max_length,
            use_social_correction, temperature
        ) 