import torch
import torch.nn as nn
import copy

import numpy as np

from typing import List, Optional, Tuple
from dataclasses import dataclass

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from transformers.modeling_utils import Conv1D, PreTrainedModel, GenerationMixin
from transformers.file_utils import ModelOutput, add_start_docstrings, add_code_sample_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from transformers.modeling_outputs import ModelOutput

from transformers import GPT2LMHeadModel

from embeddings import BinaryPositionEmbedding
from moe import MoEACTBlock


_CHECKPOINT_FOR_DOC = "gpt2act"
_CONFIG_FOR_DOC = "GPT2ACTConfig"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

class GPT2ACTConfig(GPT2Config):
    def __init__(self, act_commitment_cost=1e-3, gradient_checkpointing=False, halting_function_spec=None, layerwise_attn="simple",
                 local_window_size=None, use_relative_position=False, dynamic_stride=None, act_depth=None,
                 teacher=None, lambda_kd=1e-4, temperature_kd=4.0, use_binary_embedding=False, 
                 num_experts=4,
                 experts_top_k=2,
                 expert_capacity=None,
                 router_jitter_noise=0.0,
                 **kwargs):
        """
        :class:`~transformers.GPT2ACTConfig` is the configuration class to store the configuration of a
        :class:`~transformers.GPT2ACTModel`.

        Args:
            act_commitment_cost (:obj:`float`, `optional`, defaults to 1e-3):   The cost of the commitment ACT loss.
            gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):   If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
            halting_function_spec (:obj:`str`, `optional`, defaults to :obj:`None`):   The specification of the halting function. If :obj:`None`, the ACTLinearHaltingFunction function is not used.
            layerwise_attn (:obj:`string`, `optional`, defaults to :obj:`simple`):   If not :obj:`None`, use layerwise attention, 'simple' - simple attention, mha - MultiHeadedAttention.
            local_window_size (:obj:`int`, `optional`, defaults to :obj:`None`):   The size of the local window. If :obj:`None`, the global attention is used.
            use_relative_position (:obj:`bool`, `optional`, defaults to :obj:`False`):   If :obj:`True`, use relative position embedding.
            dynamic_stride (:obj:`bool`, `optional`, defaults to :obj:`None`):   If :obj: is a number, use dynamic stride.
            lambda_kd (:obj:`float`, `optional`, defaults to 1e-4):   The weight of the distillation loss.
            temperature_kd (:obj:`float`, `optional`, defaults to 4.0):   The temperature_kd for distillation.
            teacher (:obj:`GPT2LMHeadModel`, `optional`, defaults to :obj:`None`):   The teacher model for distillation.
            act_depth (:obj:`int`, :obj:`float`, `optional`, defaults to :obj:`None`):   The depth of the ACT model,  if :obj:`NOne`, use n_embed as depth of ACT blocks
            use_binary_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):   If :obj:`True`, use binary embedding.
            num_experts (:obj:`int`, `optional`, defaults to 4):   The number of experts.
            experts_top_k (:obj:`int`, `optional`, defaults to 2):   The number of top experts.
            expert_capacity (:obj:`int`, `optional`, defaults to :obj:`None`):   The capacity of each expert.
            router_jitter_noise (:obj:`float`, `optional`, defaults to 0.0):   The jitter noise of the router.
            kwargs (:obj:`Dict[str, any]`):   Remaining dictionary of keyword arguments from GPT2Config. 
        """
        self.act_commitment_cost = act_commitment_cost
        self.gradient_checkpointing = gradient_checkpointing
        self.halting_function_spec = halting_function_spec
        self.layerwise_attn = layerwise_attn
        self.local_window_size = local_window_size
        self.use_relative_position = use_relative_position
        self.use_binary_embedding = use_binary_embedding
        self.dynamic_stride = dynamic_stride
        self.lambda_kd = lambda_kd
        self.temperature_kd = temperature_kd
        self.teacher = teacher
        self.act_depth = act_depth
        self.num_experts = num_experts
        self.experts_top_k = experts_top_k
        self.expert_capacity = expert_capacity
        self.router_jitter_noise = router_jitter_noise

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

        if config.use_binary_embedding:
            self.wte = BinaryPositionEmbedding(config.vocab_size, config.n_embd)
            self.wpe = BinaryPositionEmbedding(config.n_positions, config.n_embd)
        else:
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Word Embedding 
            self.wpe = nn.Embedding(config.n_positions, config.n_embd) # Position Embedding

        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        if config.act_depth is not None:
            if isinstance(config.act_depth, int):
                act_depth = config.act_depth
            elif isinstance(config.act_depth, float):
                act_depth = int(config.n_embd * config.act_depth)
            else:
                raise ValueError("config.act_depth must be int or float")

            self.act_in = nn.Linear(config.n_embd, act_depth)
            self.act_out = nn.Linear(act_depth, config.n_embd)
            config = copy.deepcopy(config)
            config.n_embd = act_depth
        else:
            self.act_in = None
            self.act_out = None

        self.act_f = MoEACTBlock(
            GPT2Block(config), 
            config.n_layer, 
            config.n_embd, 
            num_experts=config.num_experts,  # New config parameter
            top_k=config.experts_top_k,  # New config parameter
            act_commitment_cost=act_commitment_cost,
            gradient_checkpointing=gradient_checkpointing, 
            dynamic_stride=config.dynamic_stride, 
            layerwise_attn=config.layerwise_attn
        )
        
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
        self.first_device="cpu"
        self.last_device="cpu"
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

        if self.act_in is not None:
            hidden_states = self.act_in(hidden_states)

        outputs = self.act_f(hidden_states,
                                past_key_values=past_key_values,
                                head_mask=head_mask,
                                attention_mask=attention_mask,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                use_cache=use_cache,
                                output_attentions=output_attentions)

        if len(outputs) == 7:
            hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions, act_loss, ponder_cost  = outputs

            if self.act_out is not None:
                hidden_states = self.act_out(hidden_states)

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
        elif len(outputs) == 3:
            combined_output, act_loss, ponder_cost = outputs
            # Properly format output for Hugging Face transformers compatibility
            if not return_dict:
                return tuple(v for v in [combined_output, None, None, None, None, act_loss, ponder_cost] if v is not None)
            
            return ACTModelOutputWithPastAndCrossAttentions(
                last_hidden_state=combined_output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
                act_loss=act_loss,
                ponder_cost=ponder_cost,
            )
        else:
            raise ValueError(f"Unexpected number of outputs: {len(outputs)}")
    
class GPT2ACTLMHeadModel(GPT2ACTPreTrainedModel, GenerationMixin):
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

    def copyweights(self, pretrained=None, model=None, freeze=False):
        assert pretrained is not None or model is not None, "You must provide a pretrained model or a model to copy the weights from"

        if model is None:
            model = GPT2LMHeadModel.from_pretrained(pretrained)
            
        self.lm_head.load_state_dict(model.lm_head.state_dict())
        self.transformer.wte.load_state_dict(model.transformer.wte.state_dict())
        self.transformer.wpe.load_state_dict(model.transformer.wpe.state_dict())
        self.transformer.ln_f.load_state_dict(model.transformer.ln_f.state_dict())

        if freeze:
            for p in [*self.lm_head.parameters(), *self.transformer.wte.parameters(), *self.transformer.wpe.parameters(), *self.transformer.ln_f.parameters()]:
                p.requires_grad = False

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    def save_pretrained(self, save_directory, safe_serialization=False, **kwargs):
        """
        Save a model with its configuration file to a directory, so that it can be re-loaded using the
        `from_pretrained` class method.
        
        Arguments:
            save_directory (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
                Set to False to address shared tensor issue between lm_head.weight and transformer.wte.weight.
            **kwargs:
                Additional key word arguments passed along to the `push_to_hub` method.
        """
        # Setting safe_serialization=False to fix the error with shared tensors
        return super().save_pretrained(save_directory, safe_serialization=False, **kwargs)

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
        loss_fct = None

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
    def __init__(self, config):

        super().__init__(config)

        self.teacher = GPT2LMHeadModel.from_pretrained(config.teacher)
        self.student = GPT2ACTLMHeadModel(config)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False


        self._lambda_kd = config.lambda_kd
        self._temperature_kd = config.temperature_kd

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

    def copyweights(self, pretrained, freeze=False):
        # initialize student with teacher's weights
        self.student.copyweights(pretrained, model=self.teacher, freeze=freeze)
        self.student.transformer.act_f._block.load_state_dict(self.teacher.transformer.h[0].state_dict())


    def to_device(self, device="cpu", **kwargs):
        """ Move teacher and student to device via to() method """
        self.teacher.to(device, **kwargs)
        self.student.to(device, **kwargs)
        
        if self.model_parallel:
            return {k: kwargs[k].to(device) if kwargs[k] is not None else None for k in kwargs}
        else:
            return kwargs
        
    def save_pretrained(self, save_directory, safe_serialization=False, **kwargs):
        """
        Save a model with its configuration file to a directory, so that it can be re-loaded using the
        `from_pretrained` class method.
        
        Arguments:
            save_directory (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
                Set to False to address shared tensor issue between lm_head.weight and transformer.wte.weight.
            **kwargs:
                Additional key word arguments passed along to the `push_to_hub` method.
        """
        # Setting safe_serialization=False to fix the error with shared tensors
        return self.student.save_pretrained(save_directory, safe_serialization=False, **kwargs)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
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
            #print('teacher_outputs.loss', teacher_outputs.loss)

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

            kd_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student_outputs.logits/self._temperature_kd, dim=1),
                                torch.nn.functional.softmax(teacher_outputs.logits/self._temperature_kd, dim=1),
                                reduction='batchmean') * self._temperature_kd**2 * self._lambda_kd

            #print('train: kd_loss', kd_loss)
            student_outputs.loss = student_outputs.loss + kd_loss
            #print('train: student_outputs.loss', student_outputs.loss)

            return student_outputs

        else:
            #print('validate: student_outputs.loss', student_outputs.loss)
            return student_outputs
