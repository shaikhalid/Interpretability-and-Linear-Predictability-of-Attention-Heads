import os
import torch
import copy
import math
import pickle
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, StaticCache
import numpy as np
import matplotlib.pyplot as plt
from models.calibrate import Calibrate
import seaborn as sns
import plotly.graph_objects as go
import random
from transformers.modeling_flash_attention_utils import _flash_attention_forward
BETA = float(os.environ.get("BETA", 0.10))

# ==============================================================================
# SERIALIZED KV CACHE DATA
# ==============================================================================
# {request_id : ,
# num_decode_tokens:,
# layer_id:
# head_id:
# K_matrix :,
# V_matrix,”
# Q_matrix,:
# input_token_ids:,
# }

batch_id = 0
request_id = 0
num_decode_tokens = 0
all_data_to_serialize = []
number_of_tokens = 0

def serialize_kv_cache():
    if Calibrate.total_tokens_collected > Calibrate.token_limit:
        return
    
    global batch_id
    global all_data_to_serialize

    
    if Calibrate.matrix_to_pickle == "k":
        kv_cache_pickle_file = f'kv_pickles/k_cache_data_{Calibrate.dataset}_{Calibrate.model_alias}.pkl'
    elif Calibrate.matrix_to_pickle == "v":
        kv_cache_pickle_file = f'kv_pickles/v_cache_data_{Calibrate.dataset}_{Calibrate.model_alias}.pkl'
    elif Calibrate.matrix_to_pickle == "q":
        kv_cache_pickle_file = f'kv_pickles/q_cache_data_{Calibrate.dataset}_{Calibrate.model_alias}.pkl'
    else:
        raise ValueError(f"Invalid matrix to pickle: {Calibrate.matrix_to_pickle}")
    
    # Append new data directly to the pickle file
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


# ==============================================================================
def llama3_atten_aug_forward_collect(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    input_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states) #
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)


    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Check if this layer should use compressed KV cache
    is_prefill = q_len != 1
    # ==============================================================================
    # SERIALIZE KV CACHE DATA
    # ==============================================================================
    
    global request_id
    global batch_id
    global num_decode_tokens
    global all_data_to_serialize
    batch_size = key_states.shape[0]
    if Calibrate.get_pickle_kv_cache():
        if is_prefill and self.layer_idx == 0:
            Calibrate.total_tokens_collected += q_len
            print("Prefill Stage for batch", batch_id, "with batch size", batch_size)
            print(Calibrate.tokenizer.batch_decode(input_ids))
            batch_id += 1
            num_decode_tokens = 0
            serialize_kv_cache() # Serialize any remaining data from previous decode phase
            all_data_to_serialize = []
        elif not is_prefill and self.layer_idx == 0: # Check for decode phase
            num_decode_tokens += 1
            # Serialize periodically during decode (e.g., every 10 tokens)
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
                
                k_matrix = key_states[req_id, head:head+1]
                v_matrix = value_states[req_id, head:head+1]
                q_matrix = query_states[req_id, head:head+1]
                
                # input_token_ids = input_ids[req_id] # Keep commented unless needed
                
                # Save in original precision
                if Calibrate.matrix_to_pickle == "k":
                    k_np = k_matrix.detach().cpu().numpy()
                    if k_np.size == 0:
                        print(f"Error: Collected empty k_matrix for batch {batch_id}, request {req_id}, layer {layer_id}, head {head_id}, shape {k_np.shape}")
                    # print(f"Standard Attention - K matrix dtype: {k_matrix.dtype}, shape: {k_matrix.shape}")
                    to_serialize = {
                        'batch_id': batch_id,
                        'request_id': req_id,
                        'num_decode_tokens': num_decode_tokens,
                        'layer_id': layer_id,
                        'head_id': head_id,
                        'k_matrix': k_np,
                    }
                elif Calibrate.matrix_to_pickle == "v":
                    v_np = v_matrix.detach().cpu().numpy()
                    if v_np.size == 0:
                        print(f"Error: Collected empty v_matrix for batch {batch_id}, request {req_id}, layer {layer_id}, head {head_id}, shape {v_np.shape}")
                    to_serialize = {
                        'batch_id': batch_id,
                        'request_id': req_id,
                        'num_decode_tokens': num_decode_tokens,
                        'layer_id': layer_id,
                        'head_id': head_id,
                        'v_matrix': v_np,
                    }
                elif Calibrate.matrix_to_pickle == "q":
                    q_np = q_matrix.detach().cpu().numpy()
                    if q_np.size == 0:
                        print(f"Error: Collected empty q_matrix for batch {batch_id}, request {req_id}, layer {layer_id}, head {head_id}, shape {q_np.shape}")
                    to_serialize = {
                        'batch_id': batch_id,
                        'request_id': req_id,
                        'num_decode_tokens': num_decode_tokens,
                        'layer_id': layer_id,
                        'head_id': head_id,
                        'q_matrix': q_np,
                    }
                all_data_to_serialize.append(to_serialize)
            

    # Handle past key value and shared KV cache
    if past_key_value is not None:
        # Original KV cache update
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )


    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

# ==============================================================================


def llama3_flash_atten_aug_forward_collect(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,          # ❶ added – needed to tag pre-/decode
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    # ----------------------------- original Flash-Attention guards -----------------------------
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2`"
        )
    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    # ----------------------------- projection --------------------------------------------------
    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,        self.head_dim).transpose(1, 2)
    key_states   = key_states.view  (bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # ----------------------------- rotary ------------------------------------------------------
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # ==========================================================================================
    # ❷  KV-CACHE COLLECTION  (identical to eager `*_collect` implementation)
    # ==========================================================================================
    global batch_id, request_id, num_decode_tokens, all_data_to_serialize, number_of_tokens

    is_prefill = q_len != 1  # same heuristic
    batch_size = key_states.shape[0]

    if Calibrate.get_pickle_kv_cache():
        # --- stage switches (prefill vs decode) ---
        number_of_tokens += q_len
        if is_prefill and self.layer_idx == 0:
            print("Prefill Stage for batch", batch_id, "with batch size", batch_size)
            batch_id += 1
            num_decode_tokens = 0
            serialize_kv_cache()          # flush leftovers
            all_data_to_serialize = []
        elif (not is_prefill) and self.layer_idx == 0:
            num_decode_tokens += 1
            serialize_kv_cache()          # periodic flush during decode

        # --- collect head matrices ---
        if Calibrate.get_pickle_kv_cache():
            layer_id = self.layer_idx
            if Calibrate.matrix_to_pickle == "k":
                num_heads = key_states.shape[1]
            elif Calibrate.matrix_to_pickle == "v":
                num_heads = value_states.shape[1]
            elif Calibrate.matrix_to_pickle == "q":
                num_heads = query_states.shape[1]
            else:
                num_heads = 0  # safety

            for head in range(num_heads):
                for req_idx in range(batch_size):
                    if Calibrate.matrix_to_pickle == "k":
                        matrix_np = key_states[req_idx, head:head+1].detach().cpu().numpy()
                        field     = "k_matrix"
                        # print(f"Flash Attention - K matrix dtype: {key_states.dtype}, shape: {key_states[req_idx, head:head+1].shape}")
                    elif Calibrate.matrix_to_pickle == "v":
                        matrix_np = value_states[req_idx, head:head+1].detach().cpu().numpy()
                        field     = "v_matrix"
                        print(f"Flash Attention - V matrix dtype: {value_states.dtype}, shape: {value_states[req_idx, head:head+1].shape}")
                    else:  # "q"
                        matrix_np = query_states[req_idx, head:head+1].detach().cpu().numpy()
                        field     = "q_matrix"
                        print(f"Flash Attention - Q matrix dtype: {query_states.dtype}, shape: {query_states[req_idx, head:head+1].shape}")

                    all_data_to_serialize.append(
                        {
                            'batch_id': batch_id,
                            'request_id': req_idx,
                            'num_decode_tokens': num_decode_tokens,
                            'layer_id': layer_id,
                            'head_id': head,
                            field: matrix_np,
                        }
                    )

    # ----------------------------- KV-cache update ---------------------------------------------
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # ----------------------------- Flash-Attention kernel (unchanged) --------------------------
    query_states = query_states.transpose(1, 2)
    key_states   = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0
    if query_states.dtype == torch.float32:
        target_dtype = (
            torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled()
            else getattr(self.config, "_pre_quantization_dtype", self.q_proj.weight.dtype)
        )
        query_states = query_states.to(target_dtype)
        key_states   = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states, key_states, value_states,
        attention_mask, q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = self.o_proj(attn_output.reshape(bsz, q_len, -1).contiguous())

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value
