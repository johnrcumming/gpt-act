import math
import torch
import torch.nn as nn

import numpy as np

from typing import List, Optional, Tuple
from dataclasses import dataclass

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from transformers.modeling_utils import Conv1D, PreTrainedModel
from transformers.file_utils import ModelOutput, add_start_docstrings, add_code_sample_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from transformers.modeling_outputs import ModelOutput

from transformers import GPT2LMHeadModel



_CHECKPOINT_FOR_DOC = "gpt2act"
_CONFIG_FOR_DOC = "GPT2ACTConfig"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

class GPT2ACTConfig(GPT2Config):
    def __init__(self, act_commitment_cost=1e-3, gradient_checkpointing=False, halting_function_spec=None, layerwise_attn=True,
                 local_window_size=None, use_relative_position=False, dynamic_stride=NOne, **kwargs):
        """
        :class:`~transformers.GPT2ACTConfig` is the configuration class to store the configuration of a
        :class:`~transformers.GPT2ACTModel`.

        Args:
            act_commitment_cost (:obj:`float`, `optional`, defaults to 1e-3):   The cost of the commitment ACT loss.
            gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):   If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
            halting_function_spec (:obj:`str`, `optional`, defaults to :obj:`None`):   The specification of the halting function. If :obj:`None`, the GPT2ACTHaltingFunction function is not used.
            layerwise_attn (:obj:`bool`, `optional`, defaults to :obj:`True`):   If :obj:`True`, use layerwise attention.
            local_window_size (:obj:`int`, `optional`, defaults to :obj:`None`):   The size of the local window. If :obj:`None`, the global attention is used.
            use_relative_position (:obj:`bool`, `optional`, defaults to :obj:`False`):   If :obj:`True`, use relative position embedding.
            dynamic_stride (:obj:`bool`, `optional`, defaults to :obj:`None`):   If :obj:`True`, use dynamic stride.
            kwargs (:obj:`Dict[str, any]`):   Remaining dictionary of keyword arguments from GPT2Config. 
        """
        self.act_commitment_cost = act_commitment_cost
        self.gradient_checkpointing = gradient_checkpointing
        self.halting_function_spec = halting_function_spec
        self.layerwise_attn = layerwise_attn
        self.local_window_size = local_window_size
        self.use_relative_position = use_relative_position
        self.dynamic_stride = dynamic_stride

        super().__init__(**kwargs)

class GPT2ACTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2ACTConfig
    base_model_prefix = "transformer"
    is_parallelizable = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class GPT2ACTRelativePositionEmbedding(nn.Module):
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

class GPT2ACTLocalAttention(nn.Module):
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
            self.relative_position_embedding = GPT2ACTRelativePositionEmbedding(2 * hidden_size - 1, hidden_size // num_heads)


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
        

class FCTBlock(nn.Module):
    def __init__(self, block, layers, hiddens, gradient_checkpointing=False, add_cross_attention=False, **kwargs):
        super().__init__()

        self._block = block    
        self._layers = layers
        self._hiddens = hiddens
        self._gradient_checkpointing = gradient_checkpointing
        self._add_cross_attention = add_cross_attention
        self._wle = nn.Embedding(layers, hiddens) # Layer Embedding

    def forward(self, hidden_states, head_mask=None, past_key_values=None, output_hidden_states=None, use_cache=None, output_attentions=None, **kwargs):
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self._add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for l in range(self._layers):
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
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

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
                outputs = self._block(hidden_states, layer_past=layer_past, head_mask=mask, use_cache=use_cache, output_attentions=output_attentions, **kwargs)
            
            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self._add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

        return [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions, 0.0, 0.0]

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
        self._layers = nn.ModuleList([block.copy() for _ in range(layers, stride)])
        
    def forward(self, hidden_states, step=1, **kwargs):
        """
        Forward pass
        
        Args:
            hidden_states: Hidden states
            step: Step
            kwargs: Additional arguments
        """
        return self._layers[step % self._stride](hidden_states, **kwargs)

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

        if dynamic_stride is not None:
            self._block = DynamicBlock(block, dynamic_stride)
        else:
            self._block = block

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
            block.attn = GPT2ACTLocalAttention(hiddens, block.attn.num_attention_heads, dropout=block.attn.attn_dropout.p, local_window_size=local_window_size, use_relative_position=use_relative_position)

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
                
            fhalting.append(nn.ReLU())
        
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

        while ((halting_probability<self._threshold) & (n_updates < self._layers)).byte().any():

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
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))
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
                outputs = self._block(hidden_states, layer_past=layer_past, head_mask=mask, use_cache=use_cache, output_attentions=output_attentions, step=step, **kwargs)
            
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


@dataclass
class ACTModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    ACT model outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        act_loss (torch.FloatTensor):
            Adaptive Computation time loss
        ponder_cost (torch.FloatTensor):
            Adaptive Computation time ponder cost
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    act_loss: Optional[torch.FloatTensor] = None
    ponder_cost: Optional[torch.FloatTensor] = None

@dataclass
class ACTCausalLMOutputWithPastAndCrossAttentions(ModelOutput):
    """
    ACT model outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        act_loss (torch.FloatTensor):
            Adaptive Computation time loss
        ponder_cost (torch.FloatTensor):
            Adaptive Computation time ponder cost
        fct_loss (torch.FloatTensor):
            loss in the transformer model
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    act_loss: Optional[torch.FloatTensor] = None
    ponder_cost: Optional[torch.FloatTensor] = None
    fct_loss: Optional[torch.FloatTensor] = None



GPT2_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPT2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Args:

    Example:

    ```python
    # On a 4 GPU machine with gpt2-large:
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


@add_start_docstrings(
    "The bare GPT2-ACT Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)        
class GPT2ACTModel(GPT2ACTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if isinstance(config, GPT2ACTConfig):
            act_commitment_cost = config.act_commitment_cost
            gradient_checkpointing = config.gradient_checkpointing
        else:
            act_commitment_cost = 1e-3
            gradient_checkpointing = False

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Word Embedding 
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # Position Embedding
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.act_f = ACTBlock(GPT2Block(config), config.n_layer, config.n_embd, act_commitment_cost=act_commitment_cost, 
                              gradient_checkpointing=gradient_checkpointing, dynamic_stride=config.dynamic_stride)
        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(2, range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, 2)
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        self.ln_f = self.ln_f.to(self.first_device)
        self.act_f = self.act_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        self.ln_f = self.ln_f.to(self.first_device)
        self.act_f = self.act_f.to(self.last_device)
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=ACTModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)
            
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.model_parallel:
            torch.cuda.set_device(self.last_device)
            hidden_states = hidden_states.to(self.last_device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)

        outputs = self.act_f(hidden_states,
                                past_key_values=past_key_values,
                                head_mask=head_mask,
                                attention_mask=attention_mask,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                use_cache=use_cache,
                                output_attentions=output_attentions)

        hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions, act_loss, ponder_cost  = outputs

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            hidden_states = hidden_states.to(self.first_device)
            act_loss = act_loss.to(self.first_device)
            ponder_cost = ponder_cost.to(self.first_device)


        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions, act_loss, ponder_cost] if v is not None)

        return ACTModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            act_loss=act_loss,
            ponder_cost=ponder_cost,
        )
    
class GPT2ACTLMHeadModel(GPT2ACTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ACTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        self.model_parallel = False
        self.device_map = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.transformer.parallelize(device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def copyweights(self, pretrained):
        model = GPT2LMHeadModel.from_pretrained(pretrained)
        self.lm_head.load_state_dict(model.lm_head.state_dict())
        self.transformer.wte.load_state_dict(model.transformer.wte.state_dict())
        self.transformer.wpe.load_state_dict(model.transformer.wpe.state_dict())

        for p in [*self.lm_head.parameters(), *self.transformer.wte.parameters(), *self.transformer.wpe.parameters()]:
            p.requires_grad = False

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=ACTCausalLMOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss_fct = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct + transformer_outputs.act_loss

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ACTCausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            act_loss=transformer_outputs.act_loss,
            ponder_cost=transformer_outputs.ponder_cost,    
            fct_loss=loss_fct      
        )




class GPT2ACTDistilation(GPT2ACTPreTrainedModel):
    def __init__(self, config, pretrained='gpt2-large', lambda_kd=1e-4, temperature=4.0, copyweights=False):
        super().__init__(config)

        self.teacher = GPT2LMHeadModel.from_pretrained(pretrained)
        self.student = GPT2ACTLMHeadModel(config)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if copyweights:
            self.student.lm_head.load_state_dict(self.teacher.lm_head.state_dict())
            self.student.transformer.wte.load_state_dict(self.teacher.transformer.wte.state_dict())
            self.student.transformer.wpe.load_state_dict(self.teacher.transformer.wpe.state_dict())

        self._lambda_kd = lambda_kd
        self._temperature = temperature

        self.model_parallel = False
        self.device_map = None
        self.first_device="cpu"
        self.last_device="cpu"


    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(2, range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, 2)
        print('parallelize: self.device_map=', self.device_map)
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.student = self.student.to(self.first_device)
        self.teacher = self.teacher.to(self.last_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        print('deparallelize')
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.student = self.student.to(self.first_device)
        self.teacher = self.teacher.to(self.last_device)
        self.model_parallel = False
        torch.cuda.empty_cache()

    def to_device(self, device="cpu", **kwargs):
        if self.model_parallel:
            return {k: kwargs[k].to(device) if kwargs[k] is not None else None for k in kwargs}
        else:
            return kwargs

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """

        # print('teacher.device', self.teacher.device)
        # print('student.device', self.student.device)
        # print('input_ids.device', input_ids.device)
        # print('labels.device', labels.device)

        if self.training:
            with torch.no_grad():
                teacher_outputs = self.teacher(**self.to_device(device=self.last_device,
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                ))

        student_outputs = self.student(**self.to_device(device=self.first_device,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ))

        if self.training:
            if self.model_parallel:
                teacher_outputs.logits = teacher_outputs.logits.to(self.first_device)

            kd_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student_outputs.logits/self._temperature, dim=1),
                                torch.nn.functional.softmax(teacher_outputs.logits/self._temperature, dim=1),
                                reduction='batchmean') * self._temperature * self._temperature
            student_outputs.loss +=  kd_loss * self._lambda_kd
            return student_outputs

        else:
            return student_outputs
