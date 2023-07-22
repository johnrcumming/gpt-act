import math
import torch
import torch.nn as nn
import copy

import numpy as np

from attention import LocalAttention, MultiHeadAttention


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
            #self._layer_attention_proj = nn.Linear(hiddens, hiddens)
            self._layerwise_mha = MultiHeadAttention(hiddens, block.attn.num_heads)

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
                kernel=""
            elif l == 'n':
                fhalting.append(nn.LayerNorm(hiddens))
            elif l == 'b':
                fhalting.append(MoveChannels(nn.BatchNorm1d(hiddens), -1, 1))
            elif l.isdigit():
                kernel = kernel + l
            elif l == 'r':   
                fhalting.append(nn.ReLU())
            elif l == 'a':
                if self._local_window_size:
                    fhalting.append(LocalAttention(hiddens, int(kernel)))
                else:
                    fhalting.append(MultiHeadAttention(hiddens, int(kernel)))
                kernel=""


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
            # projected_input = self._layer_attention_proj(hidden_states)  # Shape: [batch_size, seq_length, hidden_size]
            # attention_scores = torch.einsum('bsh,bsth->bst', projected_input, stacked_outputs)  # Shape: [batch_size, seq_length, num_layers]
            # attention_probs = torch.softmax(attention_scores, dim=-1)  # Shape: [batch_size, seq_length, num_layers]
            # weighted_outputs = torch.einsum('bst,bsth->bsh', attention_probs, stacked_outputs)  # Shape: [batch_size, seq_length, hidden_size]
            weighted_outputs = self._layerwise_mha(hidden_states)
            return [weighted_outputs, presents, all_hidden_states, all_self_attentions, all_cross_attentions, act_loss, ponder_cost]
        else:
            return [previous_state, presents, all_hidden_states, all_self_attentions, all_cross_attentions, act_loss, ponder_cost]

