import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class BinaryPositionEmbedding(nn.Module):
    def __init__(self, n_positions, d_model, debug=False):
        super(BinaryPositionEmbedding, self).__init__()

        self.n_bits = math.ceil(math.log2(n_positions))
        self.d_model = d_model
        self.embedding = nn.Embedding(self.n_bits+1, d_model)

        # Normalize Embedding Parameters
        self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1, p=2)

        # Set Weight Values for 0 to 0 and make it non-trainable
        self.embedding.weight.data[0] = 0
        self.embedding.weight.data[0].requires_grad = False

        if debug:
            # set debugging weights to binary values
            for i in range(0, self.n_bits):
                self.embedding.weight.data[i] = torch.tensor(1<<i, dtype=torch.float32)
                self.embedding.weight.data[i].requires_grad = False


    def forward(self, x):
        x_shape = x.shape

        x = x.flatten()

        # create a tensor with shape (n_bits, x.shape[0])
        bit_indices = torch.arange(self.n_bits, device=x.device).unsqueeze(1).repeat(1, x.shape[0])

        # create a tensor with shape (n_bits, x.shape[0]) where each column is the binary representation of a position
        binary_repr = (x.unsqueeze(0) & (1 << bit_indices)).T

        embeddings = torch.stack([ torch.sum(self.embedding(torch.nonzero(v).squeeze(-1)), dim=0) for v in binary_repr ])

        return embeddings
    
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



     



        
