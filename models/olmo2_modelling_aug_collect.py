from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config

from models.modelling_olmo2 import eager_attention_forward
from models.modelling_olmo2 import apply_rotary_pos_emb

import os
import pickle
import numpy as np
from models.calibrate import Calibrate # Assuming Calibrate class is here

# ==============================================================================
# SERIALIZED KV CACHE DATA
# ==============================================================================
# {request_id : ,
# num_decode_tokens:,
# layer_id:
# head_id:
# K_matrix :,
# V_matrix,"
# Q_matrix,:
# input_token_ids:,
# }
batch_id = 0
request_id = 0
num_decode_tokens = 0


all_data_to_serialize = []

def serialize_kv_cache():
    # Append new data directly to the pickle file
    if Calibrate.matrix_to_pickle == "k":
        kv_cache_pickle_file = f'kv_pickles/k_cache_data_{Calibrate.dataset}_{Calibrate.model_alias}.pkl'
    elif Calibrate.matrix_to_pickle == "v":
        kv_cache_pickle_file = f'kv_pickles/v_cache_data_{Calibrate.dataset}_{Calibrate.model_alias}.pkl'
    elif Calibrate.matrix_to_pickle == "q":
        kv_cache_pickle_file = f'kv_pickles/q_cache_data_{Calibrate.dataset}_{Calibrate.model_alias}.pkl'
    else:
        raise ValueError(f"Invalid matrix to pickle: {Calibrate.matrix_to_pickle}")
    
    global all_data_to_serialize
    global batch_id # Added for print statement context
    print(f"Appending KV cache for batch {batch_id} with {len(all_data_to_serialize)} new records.")
    if len(all_data_to_serialize) == 0:
        print("No new KV cache data to serialize.")
        return

    try:
        # Open in append binary mode
        with open(kv_cache_pickle_file, 'ab') as f:
            # Dump the list of new records as a single object
            pickle.dump(all_data_to_serialize, f)

        print(f"Successfully appended {len(all_data_to_serialize)} records to {kv_cache_pickle_file}")
        # Clear the list after successful serialization
        all_data_to_serialize = []
    except Exception as e:
        print(f"Error serializing KV cache: {e}")


def olmo2_atten_aug_forward_collect(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        batch_size, q_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ==============================================================================
        # SERIALIZE KV CACHE DATA
        # ==============================================================================
        is_prefill = q_len != 1
        
        global request_id
        global batch_id
        global num_decode_tokens
        global all_data_to_serialize
        
        if Calibrate.get_pickle_kv_cache():
            if is_prefill and self.layer_idx == 0:
                print("Prefill Stage for batch", batch_id, "with batch size", batch_size)
        
                batch_id += 1
                num_decode_tokens = 0
                serialize_kv_cache() # Serialize any remaining data from previous decode phase
                all_data_to_serialize = []
            elif not is_prefill and self.layer_idx == 0: # Check for decode phase
                num_decode_tokens += 1
                # Serialize periodically during decode
                serialize_kv_cache()
            
        if Calibrate.get_pickle_kv_cache(): # Still need this check to control overall serialization
            layer_id = self.layer_idx
            if Calibrate.matrix_to_pickle == "k":
                num_heads = key_states.shape[1]
            elif Calibrate.matrix_to_pickle == "v":
                num_heads = value_states.shape[1]
            elif Calibrate.matrix_to_pickle == "q":
                num_heads = query_states.shape[1]
            
            # Collect all data to serialize in a single list
            for head in range(num_heads):
                head_id = head
                for req_id in range(batch_size):
                    if q_len == 1:
                        k_matrix = key_states[req_id, head:head+1]
                        v_matrix = value_states[req_id, head:head+1]
                        q_matrix = query_states[req_id, head:head+1]
                    else:
                        # NO chat template so no need to adjust slicing
                        k_matrix = key_states[req_id, head:head+1]
                        v_matrix = value_states[req_id, head:head+1]
                        q_matrix = query_states[req_id, head:head+1]
                    
                    # Save in original precision
                    if Calibrate.matrix_to_pickle == "k":
                        to_serialize = {
                            'batch_id': batch_id,
                            'request_id': req_id,
                            'num_decode_tokens': num_decode_tokens,
                            'layer_id': layer_id,
                            'head_id': head_id,
                            'k_matrix': k_matrix.detach().cpu().numpy(),
                        }
                    elif Calibrate.matrix_to_pickle == "v":
                        to_serialize = {
                            'batch_id': batch_id,
                            'request_id': req_id,
                            'num_decode_tokens': num_decode_tokens,
                            'layer_id': layer_id,
                            'head_id': head_id,
                            'v_matrix': v_matrix.detach().cpu().numpy(),
                        }
                    elif Calibrate.matrix_to_pickle == "q":
                        to_serialize = {
                            'batch_id': batch_id,
                            'request_id': req_id,
                            'num_decode_tokens': num_decode_tokens,
                            'layer_id': layer_id,
                            'head_id': head_id, 
                            'q_matrix': q_matrix.detach().cpu().numpy(),
                        }
                    all_data_to_serialize.append(to_serialize)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights