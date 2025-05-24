import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.attention_mechanisms import TransformerBlock, LongTermAttention, ShortTermAttention

class IntentAwareSelfGating(nn.Module):
    """Intent-aware self-gating mechanism from DisenIDP"""
    
    def __init__(self, input_dim, intent_type):
        super(IntentAwareSelfGating, self).__init__()
        self.intent_type = intent_type  # 'I' (interest) or 'D' (dependency)
        self.W = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        """
        Args:
            X: [batch_size, seq_len, input_dim] or [user_size, input_dim]
        """
        gate = self.sigmoid(self.W(X))
        return X * gate

class ChannelAttention(nn.Module):
    """Channel attention mechanism from DisenIDP"""
    
    def __init__(self, embed_dim):
        super(ChannelAttention, self).__init__()
        self.att = nn.Parameter(torch.zeros(1, embed_dim))
        self.att_m = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self.init_weights()
        
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.att.size(1))
        self.att.data.uniform_(-stdv, stdv)
        self.att_m.data.uniform_(-stdv, stdv)
        
    def forward(self, *channel_embeddings):
        """
        Args:
            channel_embeddings: List of embeddings from different channels
        
        Returns:
            mixed_embeddings: Aggregated embeddings
            score: Attention scores
        """
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim=-1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

class DisenIDPFeatureExtractor(nn.Module):
    """Feature extraction module based on DisenIDP"""
    
    def __init__(self, user_size, embed_dim, adj_tensor=None, pretrained_embeds=None, dropout=0.2):
        super(DisenIDPFeatureExtractor, self).__init__()
        
        self.user_size = user_size
        self.embed_dim = embed_dim
        self.adj_tensor = adj_tensor  # Social network adjacency matrix
        
        # User embedding layer
        if pretrained_embeds is not None:
            self.user_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeds),
                padding_idx=0,
                freeze=False
            )
        else:
            self.user_embedding = nn.Embedding(
                user_size, 
                embed_dim,
                padding_idx=0
            )
        
        # DisenIDP's intent-aware gating
        self.interest_gate = IntentAwareSelfGating(embed_dim, "I")
        self.dependency_gate = IntentAwareSelfGating(embed_dim, "D")
        
        # Channel attention for aggregating different embeddings
        self.channel_attention = ChannelAttention(embed_dim)
        
        # Attention mechanisms for sequence modeling
        self.past_multi_att = TransformerBlock(input_size=embed_dim, n_heads=4, attn_dropout=dropout)
        self.future_multi_att = TransformerBlock(input_size=embed_dim, n_heads=4, is_FFN=False,
                                               is_future=True, attn_dropout=dropout)
        self.long_term_att = LongTermAttention(input_size=embed_dim, attn_dropout=dropout)
        self.short_term_att = ShortTermAttention(input_size=embed_dim, attn_dropout=dropout)
        
        # Sequence models
        self.past_gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
        self.past_lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
        
        # Fusion layer for combining different features
        self.fusion_layer = nn.Linear(embed_dim * 3, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _dropout_graph(self, graph, keep_prob=0.8):
        """Apply dropout to graph edges during training"""
        if not self.training:
            return graph
            
        size = graph.size()
        index = graph.coalesce().indices().t()
        values = graph.coalesce().values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
        
    def extract_social_features(self, layers=2):
        """Extract features from social network structure"""
        if self.adj_tensor is None:
            return self.user_embedding.weight
            
        # Apply dropout to graph during training
        adj = self._dropout_graph(self.adj_tensor, keep_prob=0.8)
        
        # Apply intent-aware gating to user embeddings
        u_emb_interest = self.interest_gate(self.user_embedding.weight)
        u_emb_dependency = self.dependency_gate(self.user_embedding.weight)
        
        # Collect embeddings from different layers
        all_emb_interest = [u_emb_interest]
        all_emb_dependency = [u_emb_dependency]
        
        # Graph convolution layers
        for k in range(layers):
            # Interest channel
            u_emb_interest = torch.sparse.mm(adj, u_emb_interest)
            norm_emb_interest = F.normalize(u_emb_interest, p=2, dim=1)
            all_emb_interest.append(norm_emb_interest)
            
            # Dependency channel
            u_emb_dependency = torch.sparse.mm(adj, u_emb_dependency)
            norm_emb_dependency = F.normalize(u_emb_dependency, p=2, dim=1)
            all_emb_dependency.append(norm_emb_dependency)
        
        # Aggregate embeddings from all layers
        u_emb_interest = torch.stack(all_emb_interest, dim=1)
        u_emb_interest = torch.sum(u_emb_interest, dim=1)
        
        u_emb_dependency = torch.stack(all_emb_dependency, dim=1)
        u_emb_dependency = torch.sum(u_emb_dependency, dim=1)
        
        # Aggregate channel-specific embeddings
        aggregated_embs, _ = self.channel_attention(u_emb_interest, u_emb_dependency)
        
        return aggregated_embs
        
    def extract_sequence_features(self, seq, seq_lengths, time_intervals=None):
        """Extract features from sequence data"""
        batch_size, seq_len = seq.size()
        
        # Get user embeddings
        embedded = self.user_embedding(seq)  # [batch_size, seq_len, embed_dim]
        
        # Apply intent-aware gating
        interest_emb = self.interest_gate(embedded)
        dependency_emb = self.dependency_gate(embedded)
        
        # Pack sequence for RNN processing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process with GRU
        gru_output, gru_hidden = self.past_gru(packed_embedded)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
        
        # Process with LSTM
        lstm_output, (lstm_hidden, _) = self.past_lstm(packed_embedded)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # Apply transformer attention
        transformer_output = self.past_multi_att(embedded, embedded, embedded)
        
        # Extract long-term and short-term features
        # Use the first node for long-term attention
        first_node = embedded[:, 0, :]
        # Use the last non-padding node for short-term attention
        last_indices = seq_lengths - 1
        last_indices = torch.clamp(last_indices, min=0)
        batch_indices = torch.arange(batch_size, device=seq.device)
        last_node = embedded[batch_indices, last_indices]
        
        # Apply long-term and short-term attention
        long_term_context = self.long_term_att(first_node, embedded, embedded)
        short_term_context = self.short_term_att(last_node, embedded, embedded)
        
        # Combine different features
        combined_features = torch.cat([
            gru_hidden[-1],  # Last GRU hidden state
            lstm_hidden[-1],  # Last LSTM hidden state
            transformer_output[:, -1, :]  # Last transformer output
        ], dim=-1)
        
        # Fuse features
        fused_features = self.fusion_layer(combined_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features
        
    def forward(self, seq, seq_lengths, time_intervals=None):
        """
        Forward pass for feature extraction
        
        Args:
            seq: Input sequence [batch_size, seq_len]
            seq_lengths: Sequence lengths [batch_size]
            time_intervals: Time intervals between sequence elements [batch_size, seq_len]
            
        Returns:
            features: Extracted features [batch_size, embed_dim]
        """
        # Extract sequence features
        seq_features = self.extract_sequence_features(seq, seq_lengths, time_intervals)
        
        # Extract social features if available
        if self.adj_tensor is not None:
            social_features = self.extract_social_features()
            
            # Get social features for the users in the sequence
            # Use the last user in each sequence
            last_indices = seq_lengths - 1
            last_indices = torch.clamp(last_indices, min=0)
            batch_size = seq.size(0)
            batch_indices = torch.arange(batch_size, device=seq.device)
            last_users = seq[batch_indices, last_indices]
            
            # Get social features for these users
            user_social_features = social_features[last_users]
            
            # Combine sequence and social features
            combined_features = torch.cat([seq_features, user_social_features], dim=-1)
            final_features = self.fusion_layer(combined_features)
            return final_features
        
        return seq_features 