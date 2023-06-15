# Adaptive Computation Time for Transformer Models
# based on Universal Transformer (https://arxiv.org/abs/1807.03819) and ACT (https://arxiv.org/abs/1603.08983)

import math
import copy
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

@dataclass
class ACTModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    """
    ACT model outputs, with potential hidden states and attentions.

    Args:
        act_loss (torch.FloatTensor):
            Adaptive Computation time loss
        ponder_cost (torch.FloatTensor):
            Adaptive Computation time ponder cost
    """

    act_loss: Optional[torch.FloatTensor] = None
    ponder_cost: Optional[torch.FloatTensor] = None

@dataclass
class ACTCausalLMOutputWithPastAndCrossAttentions(CausalLMOutputWithCrossAttentions):
    """
    ACT model outputs, with potential hidden states and attentions.

    Args:
        act_loss (torch.FloatTensor):
            Adaptive Computation time loss
        ponder_cost (torch.FloatTensor):
            Adaptive Computation time ponder cost
        fct_loss (torch.FloatTensor):
            loss in the transformer model
    """
    act_loss: Optional[torch.FloatTensor] = None
    ponder_cost: Optional[torch.FloatTensor] = None
    fct_loss: Optional[torch.FloatTensor] = None


class RelativePositionEmbedding(nn.Module):
    """
    This module produces relative position embeddings given a position index tensor.
    
    Args:
        max_position_embeddings (:obj:`int`):  The maximum value of the dimensionality of position embeddings, i.e. :obj:`max_position_embeddings` = :obj:`seq_length + 1`.
        hidden_size (:obj:`int`):  The hidden size of the embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self._embedding = nn.Embedding(2 * max_position_embeddings - 1, hidden_size)

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

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

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
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

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


class ACTHaltingFunction(nn.Module):
    """
    This module implements the ACT halting function as specified in the Universal Transformer.
    
    Args:
        hiddens (:obj:`int`):  The hidden size of the attention.
        local_kernel_size (:obj:`int`, `optional`, defaults to 3):  The kernel size of the local convolution.
        global_kernel_size (:obj:`int`, `optional`, defaults to 1):  The kernel size of the global convolution.
        local_bias_init (:obj:`float`, `optional`, defaults to -3.0):  The initial value of the local bias.
        local_padding_size (:obj:`int`, `optional`, defaults to 1):  The padding size of the local convolution.
    """
    def __init__(self, hiddens, local_kernel_size=3, global_kernel_size=1, local_bias_init=-3.0, local_padding_size=1):
        super().__init__()

        self._local_bn = nn.BatchNorm1d(hiddens)
        self._local_conv = nn.Conv1d(hiddens, 1, local_kernel_size, padding=local_padding_size)
        self._local_conv.bias.data.fill_(local_bias_init)
        self._global_bn = nn.BatchNorm1d(hiddens)
        self._global_conv = nn.Conv1d(hiddens, 1, global_kernel_size)
        self._sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Move channels from end
        x = x.permute(0,2,1)
        
        local_features = self._local_bn(x)
        halting_logit = self._local_conv(local_features)

        global_feature = self._global_bn(x)
        global_feature = torch.mean(global_feature, 2, keepdims=True)
        halting_logit_global = self._global_conv(global_feature)

        halting_logit = halting_logit + halting_logit_global
        halting_prob = self._sigmoid(halting_logit)

        # move channels to end
        halting_prob = halting_prob.permute(0,2,1)

        return halting_prob

class ACTLinearHaltingFunction(nn.Module):
    """
    This module implements the ACT halting function as specified by Graves et al. (2016).
    
    Args:
        hiddens (:obj:`int`):  The hidden size of the attention.
        local_kernel_size (:obj:`int`, `optional`, defaults to 3):  The kernel size of the local convolution.
        global_kernel_size (:obj:`int`, `optional`, defaults to 1):  The kernel size of the global convolution.
        local_bias_init (:obj:`float`, `optional`, defaults to -3.0):  The initial value of the local bias.
        local_padding_size (:obj:`int`, `optional`, defaults to 1):  The padding size of the local convolution.
    """
    def __init__(self, hiddens, global_kernel_size=1, local_kernel_size=3, local_bias_init=-3.0, local_padding_size=1):
        super().__init__()

        self._local_bn = nn.LayerNorm(hiddens) 
        self._local_fc = nn.Linear(hiddens, 1)
        self._local_conv = MoveChannels(nn.Conv1d(hiddens, hiddens, local_kernel_size, padding=local_padding_size),-1,1)
        self._local_conv.bias.data.fill_(local_bias_init)
        self._global_bn = nn.LayerNorm(hiddens)
        self._global_fc = nn.Linear(hiddens, 1)
        self._global_conv = MoveChannels(nn.Conv1d(hiddens, hiddens, global_kernel_size),-1,1)
        self._sigmoid = nn.Sigmoid()


    def forward(self, x):
        local_features = self._local_bn(x)
        local_features = self._local_conv(local_features)
        halting_logit = self._local_fc(local_features)

        global_feature = self._global_bn(x)
        global_feature = self._global_conv(global_feature)
        global_feature = torch.mean(global_feature, 1, keepdims=True)
        halting_logit_global = self._global_fc(global_feature)

        halting_logit = halting_logit + halting_logit_global
        halting_prob = self._sigmoid(halting_logit)

        return halting_prob

class MoveChannels(nn.Module):
    def __init__(self, block, chin=2, chblock=1):
        super().__init__()    
        self._block = block
        self._chin = chin
        self._chblock = chblock
        
    def forward(self, x):
        x = torch.transpose(x, self._chin, self._chblock)
        x = self._block(x)
        x = torch.transpose(x, self._chblock, self._chin)
        return x
    
    @property
    def bias(self):
        return self._block.bias
        
class DynamicBlock(nn.Module):
    """"
    Dynamic Block for ACTTransformer model
    
    Args:
        block: Transformer block
        stride: Stride
    """
    def __init__(self, block, layers=1, stride=1):
        super().__init__()

        self._stride = stride
        self._layers = nn.ModuleList([copy.deepcopy(block) for _ in range(layers//stride)])
        
    def forward(self, hidden_states, step=1, **kwargs):
        """
        Forward pass
        
        Args:
            hidden_states: Hidden states
            step: Step
            kwargs: Additional arguments
        """
        return self._layers[step // self._stride](hidden_states, **kwargs)


class FCTBlock(nn.Module):
    def __init__(self, block, layers, hiddens, gradient_checkpointing=False, add_cross_attention=False, 
                 local_window_size=None, use_relative_position=False, dynamic_stride=None, **kwargs):
        super().__init__()

        self._layers = layers
        self._hiddens = hiddens
        self._gradient_checkpointing = gradient_checkpointing
        self._add_cross_attention = add_cross_attention
        self._use_relative_position = use_relative_position

        self._wle = nn.Embedding(layers, hiddens) # Layer Embedding

        if local_window_size:
            block.attn = LocalAttention(hiddens, block.attn.num_attention_heads, dropout=block.attn.attn_dropout.p, local_window_size=local_window_size, use_relative_position=use_relative_position)

        if dynamic_stride is not None:
            self._block = DynamicBlock(block, layers=layers, stride=dynamic_stride)
        else:
            self._block = block


    def forward(self, hidden_states, head_mask=None, past_key_values=None, output_hidden_states=None, use_cache=None, output_attentions=None, **kwargs):
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self._add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for step in range(self._layers):
            # Add layer embedding
            s = torch.ones(hidden_states.shape[0:-1], dtype=torch.long, device=hidden_states.device) * step
            hidden_states = hidden_states + self._wle(s)

            layer_past=past_key_values[min(step, len(past_key_values)-1)] if past_key_values is not None else None

            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)

            mask = head_mask[step] if head_mask is not None else None
            # apply layer fn to state and enc_hidden
            if self._gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        if isinstance(module, DynamicBlock):
                            return tuple(output for output in module(*inputs, use_cache, output_attentions))
                        else:
                            return tuple(output for output in module(*inputs, use_cache, output_attentions, step=step))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self._block),
                    hidden_states,
                    None,
                    kwargs['attention_mask'] if 'attention_mask' in kwargs else None,
                    mask,
                    kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs else None,
                    kwargs['encoder_attention_mask'] if 'encoder_attention_mask' in kwargs else None
                )
            else:
                if isinstance(self._block, DynamicBlock):
                    outputs = self._block(hidden_states, layer_past=layer_past, head_mask=mask, use_cache=use_cache, output_attentions=output_attentions, step=step, **kwargs)
                else:
                    outputs = self._block(hidden_states, layer_past=layer_past, head_mask=mask, use_cache=use_cache, output_attentions=output_attentions, **kwargs)
            
            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self._add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

        return [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions, 0.0, 0.0]


class ACTBlock(nn.Module):
    """
    ACT Block for ACTTransformer model
    
    Args:
        block: Transformer block
        layers: Number of layers
        hiddens: Hidden size
        initial_halting_bias: Initial bias for halting probability
        act_commitment_cost: ACT commitment cost
        layer_penalty: Layer penalty
        epsilon: Epsilon for halting probability
        gradient_checkpointing: Use gradient checkpointing
        add_cross_attention: Add cross attention
        halting_function_spec: Halting function specification
        layerwise_attn: Use layerwise attention
        local_window_size: Local window size
        use_relative_position: Use relative position encoding
        dynamic_stride: Use dynamic stride   
    """
    def __init__(self, block, layers, hiddens, initial_halting_bias=-1, act_commitment_cost=1e-3, layer_penalty=1e-3, epsilon=1e-2, 
                 gradient_checkpointing=False, add_cross_attention=False, halting_function_spec=None, layerwise_attn=True,
                 local_window_size=None, use_relative_position=False, dynamic_stride=None):
        super().__init__()

        self._layers = layers
        self._hiddens = hiddens
        self._layerwise_attn = layerwise_attn
        self._local_window_size=local_window_size
        self._use_relative_position = use_relative_position

        self._threshold = 1 - epsilon
        self._act_commitment_cost = act_commitment_cost
        self._layer_penalty = layer_penalty # Tau - Graves 2016
        self._gradient_checkpointing = gradient_checkpointing
        self._add_cross_attention = add_cross_attention
        self._wle = nn.Embedding(layers, hiddens) # Layer Embedding

        if self._layerwise_attn:
            self._layer_attention_proj = nn.Linear(hiddens, hiddens)

        if halting_function_spec is not None:
            self._Fhalting = self.make_halting_function(halting_function_spec, hiddens)  
            torch.nn.init .constant_(self._Fhalting[0].bias, initial_halting_bias)
        else:
            self._Fhalting = ACTLinearHaltingFunction(hiddens)

        if local_window_size:
            block.attn = LocalAttention(hiddens, block.attn.num_attention_heads, dropout=block.attn.attn_dropout.p, local_window_size=local_window_size, use_relative_position=use_relative_position)

        if dynamic_stride is not None:
            self._block = DynamicBlock(block, layers=layers, stride=dynamic_stride)
        else:
            self._block = block

    def make_halting_function(self, halting_function, hiddens):
        fhalting = []
        kernel = ""
        for l in halting_function[:-1]:
            if l == 'l':
                fhalting.append(nn.Linear(hiddens,hiddens))
            elif l == 'c':
                fhalting.append(MoveChannels(nn.Conv1d(hiddens,hiddens, int(kernel) if kernel.isdigit() else 1), -1, 1))
            elif l == 'n':
                fhalting.append(nn.LayerNorm(hiddens))
            elif l == 'b':
                fhalting.append(MoveChannels(nn.BatchNorm1d(hiddens), -1, 1))
            elif l.isdigit():
                kernel = kernel + l
            elif l == 'r':   
                fhalting.append(nn.ReLU())
            elif l == 'a':
                fhalting.append(LocalAttention(hiddens, kernel))
        
        if halting_function[-1] == 'l':
            fhalting.append(nn.Linear(hiddens,1))    
        elif halting_function[-1] == 'c':
            fhalting.append(MoveChannels(nn.Conv1d(hiddens,1,1), -1, 1))
        fhalting.append(nn.Sigmoid())
        return nn.Sequential(*fhalting)

    def forward(self, hidden_states, head_mask=None, past_key_values=None, output_hidden_states=None, use_cache=None, output_attentions=None, **kwargs):
        halting_probability = torch.zeros(hidden_states.shape[0:-1], dtype=hidden_states.dtype, device=hidden_states.device)
        remainders = torch.zeros(hidden_states.shape[0:-1], dtype=hidden_states.dtype, device=hidden_states.device)
        n_updates = torch.zeros(hidden_states.shape[0:-1], dtype=hidden_states.dtype, device=hidden_states.device)
        previous_state = torch.zeros_like(hidden_states)
        step = 0
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self._add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        
        if self._layerwise_attn:
            layer_outputs = []

        while ((halting_probability<self._threshold) & (n_updates < self._layers)).bool().any():

            # Add layer signal
            s = torch.ones(hidden_states.shape[0:-1], dtype=torch.long, device=hidden_states.device) * step
            hidden_states = hidden_states + self._wle(s)

            p = self._Fhalting(hidden_states).squeeze(-1)
            # Mask for inputs hich have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self._threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self._threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1.0 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            layer_past=past_key_values[min(step, len(past_key_values)-1)] if past_key_values is not None else None

            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)

            mask = head_mask[step] if head_mask is not None else None
            # apply layer fn to state and enc_hidden
            if self._gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        if isinstance(module, DynamicBlock):
                            return tuple(output for output in module(*inputs, use_cache, output_attentions))
                        else:
                            return tuple(output for output in module(*inputs, use_cache, output_attentions, step=step))
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self._block),
                    hidden_states,
                    None,
                    kwargs['attention_mask'] if 'attention_mask' in kwargs else None,
                    mask,
                    kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs else None,
                    kwargs['encoder_attention_mask'] if 'encoder_attention_mask' in kwargs else None,
                    
                )

            else:
                if isinstance(self._block, DynamicBlock):
                    outputs = self._block(hidden_states, layer_past=layer_past, head_mask=mask, use_cache=use_cache, output_attentions=output_attentions, step=step, **kwargs)
                else:
                    outputs = self._block(hidden_states, layer_past=layer_past, head_mask=mask, use_cache=use_cache, output_attentions=output_attentions, **kwargs)

            hidden_states, present = outputs[:2]

            if self._layerwise_attn:
                layer_outputs.append(hidden_states)

            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self._add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)
                
            # update running part in the weighted state and keep the rest
            previous_state = ((hidden_states * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of the loop 
            step+=1
        
        ponder_cost = -1 * self._layer_penalty * torch.mean(step + remainders)  
        p_t = remainders + n_updates
        p_t_dims = (0, *range(1,len(p_t.shape)-1))
        p_t_size = np.prod([p_t.shape[d] for d in p_t_dims])
        act_loss = self._act_commitment_cost * torch.sum(torch.sum(p_t,dim=p_t_dims)/p_t_size/p_t.shape[0])

        stacked_outputs = torch.stack(layer_outputs, dim=2)  # Shape: [batch_size, seq_length, num_layers, hidden_size]

        if self._layerwise_attn:
            projected_input = self._layer_attention_proj(hidden_states)  # Shape: [batch_size, seq_length, hidden_size]
            attention_scores = torch.einsum('bsh,bsth->bst', projected_input, stacked_outputs)  # Shape: [batch_size, seq_length, num_layers]
            attention_probs = torch.softmax(attention_scores, dim=-1)  # Shape: [batch_size, seq_length, num_layers]
            weighted_outputs = torch.einsum('bst,bsth->bsh', attention_probs, stacked_outputs)  # Shape: [batch_size, seq_length, hidden_size]
            return [weighted_outputs, presents, all_hidden_states, all_self_attentions, all_cross_attentions, act_loss, ponder_cost]
        else:
            return [previous_state, presents, all_hidden_states, all_self_attentions, all_cross_attentions, act_loss, ponder_cost]
