import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from embeddings import RelativePositionEmbedding

class LocalAttention(nn.Module):
    """
    This module implements local attention.
    Args:
        hidden_size (:obj:`int`):  The hidden size of the attention.
        num_heads (:obj:`int`):  The number of attention heads.
        dropout (:obj:`float`, `optional`, defaults to 0.1):  The dropout probability.
        local_window_size (:obj:`int`, `optional`, defaults to :obj:`None`):  The size of the local window. If :obj:`None`, the global attention is used.
        use_relative_position (:obj:`bool`, `optional`, defaults to :obj:`False`):  If :obj:`True`, use relative position embedding.
        
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1, local_window_size=None, use_relative_position=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.local_window_size = local_window_size

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.use_relative_position = use_relative_position
        if use_relative_position:
            self.relative_position_embedding = RelativePositionEmbedding(2 * hidden_size - 1, hidden_size // num_heads)


    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        """
        Args:
            hidden_states (:obj:`torch.Tensor`):  The input tensor of shape :obj:`(batch_size, seq_length, hidden_size)`.
            attention_mask (:obj:`torch.Tensor`):  The attention mask of shape :obj:`(batch_size, seq_length, seq_length)`.
            head_mask (:obj:`torch.Tensor`):  The head mask of shape :obj:`(num_heads,)`.
        Returns:
            :obj:`torch.Tensor`:  The output tensor of shape :obj:`(batch_size, seq_length, hidden_size)`.
        """
        q = self.query_linear(hidden_states)
        k = self.key_linear(hidden_states)
        v = self.value_linear(hidden_states)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.local_window_size:
            attn_scores = self._local_attention(q, k)
        else:
            attn_scores = torch.matmul(q, k.transpose(-1, -2))

        attn_scores /= math.sqrt(self.hidden_size // self.num_heads)

        if self.use_relative_position:
            attn_scores += self.relative_position_embedding(q, k)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        attn_output = torch.matmul(attn_probs, v)
        attn_output = self._merge_heads(attn_output)

        attn_output = self.out_linear(attn_output)

        return attn_output

    def _split_heads(self, x):
        batch_size, seq_length, hidden_size = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, -1)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        batch_size, num_heads, seq_length, head_size = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_length, self.hidden_size)

    def _local_attention(self, q, k):
        seq_length = q.size(-2)
        local_window_size = min(self.local_window_size, seq_length)
        attn_scores = []

        for i in range(seq_length):
            start = max(0, i - local_window_size + 1)
            end = i + 1
            local_k = k[:, :, start:end, :]
            local_attn_scores = torch.matmul(q[:, :, i:i+1, :], local_k.transpose(-1, -2))
            attn_scores.append(local_attn_scores)

        return torch.cat(attn_scores, dim=-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert hidden_size % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = hidden_size
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        batch_size = hidden_states.size(0)

        query = self.query_linear(hidden_states).view(batch_size, -1, self.num_heads, self.d_k)
        key = self.key_linear(hidden_states).view(batch_size, -1, self.num_heads, self.d_k)
        value = self.value_linear(hidden_states).view(batch_size, -1, self.num_heads, self.d_k)

        query, key, value = [x.transpose(1, 2) for x in (query, key, value)]

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)

        # Apply Scaled Dot Product Attention using einsum
        scores = torch.einsum("bnqd,bnkd->bnqk", [query, key]) / math.sqrt(self.d_k)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply Dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Apply head_mask if it is not None
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.einsum("bnqk,bnkd->bnqd", [attn_weights, value])

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out_linear(context)

        return output

