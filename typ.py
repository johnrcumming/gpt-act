# typ.py - Implemention of Transformer Weight Prediction
#

import math
import copy
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np

from transformers.file_utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers import PretrainedConfig
from transformers.modeling_utils import Conv1D, PreTrainedModel

from act import ACTLinearHaltingFunction, LocalAttention, DynamicBlock, MoveChannels

class LayerWeightPredictorConfig(PretrainedConfig):
    """Configuration class to store the configuration of a LayerWeightPredictor Transformer model.
       The Layer Weight predictor is used to predict the weight of each layer in the Transformer.

       It is implemented as a Transformer encoder with a linear layer on top as the prediction head.
    
    Args:
        n_layers (:obj:`int`, `optional`, defaults to 12):   The number of trnsformer layers in the Layer Weight Predictor.
        n_heads (:obj:`int`, `optional`, defaults to 12):   The number of heads in the multi-head attention layers in the Layer Weight Predictor.
        hidden_size (:obj:`int`, `optional`, defaults to 768):   The size of the hidden states in the Layer Weight Predictor.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):   The size of the intermediate layer in the Layer Weight Predictor.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):   The non-linear activation function (function or string) in the Layer Weight Predictor.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):   The dropout probability for all fully connected layers in the Layer Weight Predictor.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):   The dropout probability for all attention layers in the Layer Weight Predictor.
        max_layer_predictions (:obj:`int`, `optional`, defaults to 12):  The maximum number of layers that the layer weight predictor will predict weights for.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):   The epsilon used by the layer normalization layers in the Layer Weight Predictor.
        norm_first (:obj:`bool`, `optional`, defaults to :obj:`True`):   If :obj:`True`, use layer normalization before the attention layers in the Layer Weight Predictor.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):   If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
    """

    def __init__(self, n_layers=12, n_heads=12, hidden_size=768, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 max_layer_predictions=12, layer_norm_eps=1e-12, norm_first=True, gradient_checkpointing=False, **kwargs):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_layer_predictions = max_layer_predictions
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.gradient_checkpointing = gradient_checkpointing

        super().__init__(**kwargs)


class LayerWeightPredictorPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayerWeightPredictorConfig
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


class LayerWeightPredictor(LayerWeightPredictorPreTrainedModel):
    """

        This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
        library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
        etc.)

        This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
        Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
        and behavior.

        Parameters:
            config ([`LayerWeightPredictorConfig`]): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the
                configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """    
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = nn.Embedding(config.max_layer_predictions, config.hidden_size)
        transformer_encoder_layer_args = {'d_model': config.hidden_size, 
                                          'nhead': config.n_heads, 
                                          'dim_feedforward': config.intermediate_size, 
                                          'dropout': config.hidden_dropout_prob, 
                                          'activation': config.hidden_act,
                                          'layer_norm_eps': config.layer_norm_eps,
                                          'norm_first': config.norm_first}

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(**transformer_encoder_layer_args),
                                                 num_layers=config.n_layers, 
                                                 norm=nn.LayerNorm(config.hidden_size))
        self.predictor = nn.Linear(config.hidden_size, config.max_layer_predictions)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):



class TWPConfig(PretrainedConfig):
    def __init__(self, act_commitment_cost=1e-3, gradient_checkpointing=False, halting_function_spec=None, layerwise_attn=True,
                 local_window_size=None, use_relative_position=False, dynamic_stride=None, 
                 teacher=None, lambda_kd=1e-4, temperature_kd=4.0, **kwargs):
        """
        :class:`~TWPConfig` is the configuration class to store the configuration of a
        :class:`~TWPModel`.

        Args:
            act_commitment_cost (:obj:`float`, `optional`, defaults to 1e-3):   The cost of the commitment ACT loss.
            gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):   If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
            halting_function_spec (:obj:`str`, `optional`, defaults to :obj:`None`):   The specification of the halting function. If :obj:`None`, the GPT2ACTHaltingFunction function is not used.
            layerwise_attn (:obj:`bool`, `optional`, defaults to :obj:`True`):   If :obj:`True`, use layerwise attention.
            local_window_size (:obj:`int`, `optional`, defaults to :obj:`None`):   The size of the local window. If :obj:`None`, the global attention is used.
            use_relative_position (:obj:`bool`, `optional`, defaults to :obj:`False`):   If :obj:`True`, use relative position embedding.
            dynamic_stride (:obj:`bool`, `optional`, defaults to :obj:`None`):   If :obj:`True`, use dynamic stride.
            lambda_kd (:obj:`float`, `optional`, defaults to 1e-4):   The weight of the distillation loss.
            temperature_kd (:obj:`float`, `optional`, defaults to 4.0):   The temperature_kd for distillation.
            teacher (:obj:`GPT2LMHeadModel`, `optional`, defaults to :obj:`None`):   The teacher model for distillation.

            kwargs (:obj:`Dict[str, any]`):   Remaining dictionary of keyword arguments from downstream Config. 
        """
        self.act_commitment_cost = act_commitment_cost
        self.gradient_checkpointing = gradient_checkpointing
        self.halting_function_spec = halting_function_spec
        self.layerwise_attn = layerwise_attn
        self.local_window_size = local_window_size
        self.use_relative_position = use_relative_position
        self.dynamic_stride = dynamic_stride
        self.lambda_kd = lambda_kd
        self.temperature_kd = temperature_kd
        self.teacher = teacher

        super().__init__(**kwargs)


class TWPBlock(nn.Module):
    def __init__(self, block, layers, hiddens, gradient_checkpointing=False, add_cross_attention=False, 
                 local_window_size=None, use_relative_position=False, dynamic_stride=None, **kwargs):
        super().__init__()

        self._layers = layers
        self._hiddens = hiddens
        self._gradient_checkpointing = gradient_checkpointing
        self._add_cross_attention = add_cross_attention
        self._use_relative_position = use_relative_position
        self._layer_weight_predictor = 
        
        


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

