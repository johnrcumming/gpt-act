import torch
import math

class BinaryPositionEmbedding(torch.nn.Module):
    """
    This module produces binary embeddings given a position index tensor.
    Binary Embeddings are a set of learnable embeddings that are the sum of a set of binary vectors.

    Args:
        n_positions (:obj:`int`):  The maximum value of the dimensionality of position embeddings, i.e. :obj:`n_positions` = :obj:`seq_length + 1`.
        d_model (:obj:`int`):  The hidden size of the embeddings.
    """
    def __init__(self, n_positions, d_model):
        super(BinaryPositionEmbedding, self).__init__()
        
        self.n_bits = math.ceil(math.log2(n_positions))
        self.d_model = d_model
        self.d_length = 2**self.n_bits
        self.embedding = torch.nn.Embedding(self.n_bits, self.d_model)
        # Normalize Embedding Parameters
        self.embedding.weight.data = torch.nn.functional.normalize(self.embedding.weight.data, dim=1, p=2)

    def embed(self, pos):
        """
        Embed a single position index.
        Args:
            pos (:obj:`int`):  The position index to embed.
        Returns:
            :obj:`torch.Tensor`:  The embedding of the position index :obj:`pos`.
        """        
        y = torch.LongTensor([i for i in range(self.n_bits) if torch.bitwise_and(pos, 2**i) != 0]).to(self.embedding.weight.device)
        y = self.embedding(y)
        y = torch.sum(y, dim=0)
        return y
    
    def forward(self, x):
        """
        Embed a set of position indices.
        Args:
            x (:obj:`torch.Tensor`):  The position indices to embed.
        Returns:
            :obj:`torch.Tensor`:  The embeddings of the position indices :obj:`x`.
        """
        x_shape = x.shape
        x = torch.flatten(x)
        y = torch.stack([self.embed(i) for i in x])
        y = y.reshape(*x_shape, self.d_model)
        return y


class BinaryRelativePositionEmbedding(torch.nn.Module):
    """
    This module produces relative position embeddings given a position index tensor.
    
    Args:
        max_position_embeddings (:obj:`int`):  The maximum value of the dimensionality of position embeddings, i.e. :obj:`max_position_embeddings` = :obj:`seq_length + 1`.
        hidden_size (:obj:`int`):  The hidden size of the embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self._embedding = BinaryPositionEmbedding(2 * max_position_embeddings - 1, hidden_size)

    def forward(self, q, k):
        """
        Args:
            q (:obj:`torch.Tensor`):  The query tensor of shape :obj:`(batch_size, num_heads, seq_length, dim_per_head)`.
            k (:obj:`torch.Tensor`):  The key tensor of shape :obj:`(batch_size, num_heads, seq_length, dim_per_head)`.
        Returns:
            :obj:`torch.Tensor`:  The relative position embedding of shape :obj:`(batch_size, num_heads, seq_length, seq_length)`.
        """
        seq_length = q.size(-2)
        positions = torch.arange(-seq_length + 1, seq_length, device=q.device).long()
        relative_positions = self._embedding(positions)
        relative_scores = torch.einsum('bhld,md->bhlm', q, relative_positions)
        return relative_scores

class RelativePositionEmbedding(torch.nn.Module):
    """
    This module produces relative position embeddings given a position index tensor.
    
    Args:
        max_position_embeddings (:obj:`int`):  The maximum value of the dimensionality of position embeddings, i.e. :obj:`max_position_embeddings` = :obj:`seq_length + 1`.
        hidden_size (:obj:`int`):  The hidden size of the embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self._embedding = torch.nn.Embedding(2 * max_position_embeddings - 1, hidden_size)

    def forward(self, q, k):
        """
        Args:
            q (:obj:`torch.Tensor`):  The query tensor of shape :obj:`(batch_size, num_heads, seq_length, dim_per_head)`.
            k (:obj:`torch.Tensor`):  The key tensor of shape :obj:`(batch_size, num_heads, seq_length, dim_per_head)`.
        Returns:
            :obj:`torch.Tensor`:  The relative position embedding of shape :obj:`(batch_size, num_heads, seq_length, seq_length)`.
        """
        seq_length = q.size(-2)
        positions = torch.arange(-seq_length + 1, seq_length, device=q.device).long()
        relative_positions = self._embedding(positions)
        relative_scores = torch.einsum('bhld,md->bhlm', q, relative_positions)
        return relative_scores
