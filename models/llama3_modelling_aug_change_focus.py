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
from transformers.cache_utils import Cache
import numpy as np
import matplotlib.pyplot as plt
from models.calibrate import Calibrate
import seaborn as sns
import plotly.graph_objects as go

def get_log_term(n, alpha=0.3):
    positions = torch.arange(n, device='cuda', dtype=torch.float)
    # fraction runs from 1.0 at pos=0 down to ~1/n at pos=n-1
    fraction = (n - positions) / n
    # raise it to a power alpha (e.g. alpha=1 for linear decay)
    return fraction**alpha




def find_sink_tokens(attention_map, focus_head: None, ref_layer = False, input_ids= None) -> List[int]:

    focus_head = [focus_head] if focus_head is not None else range(attention_map.shape[1])
    sink_tokens = []

    if input_ids is not None:
        print(Calibrate.tokenizer.batch_decode(input_ids))

    for head in focus_head:
        # Get the attention weights for the current head
        head_attention = attention_map[0, head]
        shape = head_attention.shape[0]
        device = head_attention.device
        B, H, N, N2 = attention_map.shape
        # Calculate the sum of attention weights for each token
        log_term = get_log_term(N).to(device)            # shape [N]
        denom = (1 + log_term).view(N)
        token_sum = torch.sum(head_attention, dim=0) / (
            denom
        ).to(device)

        # Find the indices of the top 5 tokens
        top_k = 5
        sink_indices = torch.topk(token_sum, top_k, dim=0).indices
        
        # decode the token IDs to strings
        for idx in sink_indices:
            token_str = Calibrate.tokenizer.decode(input_ids[0][idx.item()]).strip()
            sink_tokens.append(token_str)
    
        if ref_layer:
            # Store sink tokens for reference layer
            for token in sink_tokens:
                Calibrate.ref_layer_sink_tokens[f'{Calibrate.reference_layer}_{head}'].add(token)
        else:
            # Store sink tokens for focus layer
            for token in sink_tokens:
                Calibrate.focus_layer_sink_tokens[f'{Calibrate.focus_layer}_{head}'].add(token)
        


def find_focus_indices(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Find the indices of the tokens that are in focus.
    """
    focus_indices = []
    for i in range(len(input_ids)):
        token_str = Calibrate.tokenizer.decode(input_ids[i]).strip() # Decode and remove leading/trailing whitespace
        if token_str.isdigit():
            focus_indices.append(i)
    focus_indices = torch.tensor(focus_indices, dtype=torch.long) # Ensure long type
    if len(focus_indices) == 0:
        # Return tensor containing -1 if no focus tokens found
        focus_indices = torch.tensor([-1], dtype=torch.long)
    # DEBUG: print the focus_indices
    return focus_indices

def change_focus(
    attention_map: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Change the focus of the attention weights. Takes beta fraction of attention
    from non-focus tokens and redistributes it proportionally to focus tokens.
    Assumes batch size is 1.
    """
    beta = 0.01
    epsilon = 1e-9 # Small value to prevent division by zero

    # Assuming attention_map shape is [batch, heads, query_len, key_len]
    if attention_map.shape[0] != 1:
        # Keeping the assumption from the original code
        raise NotImplementedError("change_focus currently only supports batch size 1")

    device = attention_map.device
    num_heads = attention_map.shape[1]
    seq_len = attention_map.shape[-1] # Key sequence length

    # Process the single batch item
    input_ids_item = input_ids[0]
    focus_idxs = find_focus_indices(input_ids_item).to(device)

    #head_num = Calibrate.focus_head
    for head_num in range(num_heads):
        modified_head = attention_map[0][head_num].clone()
        copied_attention_map = copy.deepcopy(modified_head.detach())

        # Identify non-indices (tokens we'll take attention from)
        other_indices_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        other_indices_mask[focus_idxs] = False
        
        original_weights_on_others = modified_head[:, other_indices_mask] # Shape: [seq_len, num_other_indices]
        sum_weights_on_others_per_source = original_weights_on_others.sum(dim=1) # Shape: [seq_len]

        weight_to_move_per_source = sum_weights_on_others_per_source * (1.0 - beta)

        modified_head[:, other_indices_mask] *= beta

        original_weights_to_indices = attention_map[0, head_num, :, focus_idxs].clone()
        num_indices = focus_idxs.numel()

        if num_indices > 0:
            equal_share = 1.0 / num_indices
            distribution_ratios = torch.full_like(original_weights_to_indices, equal_share) # Shape: [seq_len, num_indices]
        else: # Should be caught earlier, but for safety
                distribution_ratios = torch.zeros_like(original_weights_to_indices)

        weight_to_add = weight_to_move_per_source.unsqueeze(1) * distribution_ratios # Shape: [seq_len, num_indices]

        # Add the weight to the target columns in the modified map
        modified_head[:, focus_idxs] += weight_to_add

        # --- Normalize ---
        # Ensure each row sums to 1 after modification
        row_sums = modified_head.sum(dim=1, keepdim=True)
        # Prevent division by zero if a row sum becomes zero (e.g., beta=0 and token only attended to others)
        modified_head = modified_head / (row_sums + epsilon)

        attention_map[0, head_num, :, :] = modified_head

    # --- Return the modified attention map ---

    return attention_map
    

def llama3_atten_aug_forward_change_focus(
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
        input_ids: Optional[torch.Tensor] = None,
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
            key_states = self.k_proj(hidden_states)
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
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        ##############BEGIN:MODIFICATION##############
        # --- Plotting Logic Start ---
        # Check conditions for plotting: correct layer, focus head specified, input_ids available
        # plot_this_layer = (
        #     self.layer_idx == Calibrate.focus_layer and
        #     Calibrate.focus_head is not None and
        #     input_ids is not None and
        #     hasattr(Calibrate, 'tokenizer') # Ensure tokenizer is available
        # )

        # token_sum_before_np = None
        # decoded_tokens = None
        # N = None
        # device = None
        # denom = None

        # if plot_this_layer:
        #     try:
        #         # --- Calculate scores BEFORE modification ---
        #         # Ensure batch size is 1 for plotting as assumed elsewhere
        #         if attn_weights.shape[0] == 1:
        #             attn_weights_before = attn_weights[0, Calibrate.focus_head].clone().detach()
        #             N = attn_weights_before.shape[-1]
        #             device = attn_weights_before.device
        #             log_term = get_log_term(N).to(device)
        #             denom = (1 + log_term).view(N)
        #             token_sum_before = torch.sum(attn_weights_before, dim=0) / denom
        #             token_sum_before_np = token_sum_before.cpu().numpy()

        #             # Decode tokens
        #             decoded_tokens = [Calibrate.tokenizer.decode(token_id).strip() for token_id in input_ids[0]]
        #         else:
        #             print(f"Skipping plot for layer {self.layer_idx}: Batch size is not 1 ({attn_weights.shape[0]})")
        #             plot_this_layer = False # Disable plotting if batch size != 1

        #     except Exception as e:
        #         print(f"Error during pre-modification calculation or decoding for plotting: {e}")
        #         plot_this_layer = False # Disable plotting on error


        # --- Apply modification ---
        if self.layer_idx == Calibrate.focus_layer and Calibrate.change_focus:
            attn_weights = change_focus( # Store modified weights separately
                attn_weights, input_ids # Pass a clone to avoid modifying original if plotting fails
            )
            # Find sink tokens for the focus layer (using modified weights)

            # # --- Calculate scores AFTER modification (if plotting is enabled) ---
            # if plot_this_layer and token_sum_before_np is not None and decoded_tokens is not None and denom is not None:
            #     try:
            #         # Ensure batch size is still 1 after modification
                    
            #         attn_weights_after = attn_weights[0, Calibrate.focus_head].detach()
            #         # N and denom are the same as before
            #         token_sum_after = torch.sum(attn_weights_after, dim=0) / denom
            #         token_sum_after_np = token_sum_after.cpu().numpy()

            #         # --- Create the side-by-side plot ---
            #         fig, axes = plt.subplots(1, 2, figsize=(max(20, len(decoded_tokens) * 0.6), 7), sharey=True) # Create two subplots, share y-axis
            #         fig.suptitle(f"Attention Scores Comparison (Layer {self.layer_idx}, Head {Calibrate.focus_head})", fontsize=14)

            #         x_indices = np.arange(len(decoded_tokens))

            #         # Plot Before
            #         axes[0].plot(x_indices, token_sum_before_np, label='Before Modification', marker='o', linestyle='-')
            #         axes[0].set_title('Before Modification')
            #         axes[0].set_xlabel("Decoded Tokens")
            #         axes[0].set_ylabel("Attention Score (Sum / (1 + log_term))")
            #         axes[0].set_xticks(x_indices)
            #         axes[0].set_xticklabels(decoded_tokens, rotation=90, fontsize=8)
            #         axes[0].grid(True, axis='y')
            #         axes[0].legend()


            #         # Plot After
            #         axes[1].plot(x_indices, token_sum_after_np, label='After Modification', marker='x', linestyle='--', color='orange')
            #         axes[1].set_title('After Modification')
            #         axes[1].set_xlabel("Decoded Tokens")
            #         # axes[1].set_ylabel("Attention Score (Sum / (1 + log_term))") # Y-label is shared
            #         axes[1].set_xticks(x_indices)
            #         axes[1].set_xticklabels(decoded_tokens, rotation=90, fontsize=8)
            #         axes[1].grid(True, axis='y')
            #         axes[1].legend()


            #         plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

            #         # Save the plot
            #         plot_dir = "attention_plots"
            #         os.makedirs(plot_dir, exist_ok=True) # Create directory if it doesn't exist
            #         plot_filename = os.path.join(plot_dir, f"attention_scores_comparison_layer_{self.layer_idx}_head_{Calibrate.focus_head}.png")
            #         plt.savefig(plot_filename)
            #         print(f"Saved attention score comparison plot to {plot_filename}")
            #         plt.close(fig) # Close the figure to free memory

            #         print("saved the plot successfully")
            #         # --- IMPORTANT: Assign the modified weights back ---


            #     except Exception as e:
            #         print(f"Error during post-modification calculation or plotting: {e}")
            #         if 'fig' in locals() and plt.fignum_exists(fig.number): # Check if fig exists and is open
            #              plt.close(fig) # Ensure plot is closed on error
            #         # --- IMPORTANT: Assign the modified weights back even if plotting failed ---
            #         attn_weights = attn_weights_modified
            # else:
            #      # --- IMPORTANT: Assign the modified weights back if plotting was disabled initially ---
            #      attn_weights = attn_weights_modified


        # --- Plotting Logic End ---
        ##############END:MODIFICATION##############

        if self.layer_idx == Calibrate.focus_layer:
            find_sink_tokens(attn_weights, Calibrate.focus_head, input_ids=input_ids)

        if self.layer_idx == Calibrate.reference_layer:
            find_sink_tokens(attn_weights, Calibrate.reference_head, ref_layer=True, input_ids=input_ids)


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
