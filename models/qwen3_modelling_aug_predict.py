import os
import pickle
import numpy as np
import logging
from torch import nn

from typing import Callable, Optional, Tuple, Union, List, Dict

import torch

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward
from models.calibrate import Calibrate

import json
import warnings
import copy
import math
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER FUNCTIONS FOR PREDICTION
# ==============================================================================

def _get_prediction_file_paths():
    """Get file paths for prediction dependencies and models."""
    if Calibrate.default_prediction_model:
        print("============ Using default prediction model ============")
        base_name = f"ifeval_{Calibrate.model_alias}"
    else:
        base_name = f"{Calibrate.dataset}_{Calibrate.model_alias}"
    
    return {
        'dependency_file_keys': f"ref_target_mapping/ref_target_heads_{base_name}_keys.json",
        'model_weights_file_keys': f"model_pickles/all_model_weights_{base_name}_keys.pkl",
        'dependency_file_values': f"ref_target_mapping/ref_target_heads_{base_name}_values.json",
        'model_weights_file_values': f"model_pickles/all_model_weights_{base_name}_values.pkl"
    }

def _load_dependencies(dependency_file, layer_idx):
    """Load and parse dependency file."""
    if not os.path.exists(dependency_file):
        warnings.warn(f"Layer {layer_idx}: Dependency file '{dependency_file}' not found.", UserWarning)
        return {}, {}
    
    try:
        with open(dependency_file, "r") as f:
            dependencies_raw = json.load(f)
        
        loaded_dependencies = {}
        targets_by_layer_temp = {}
        
        for key, value in dependencies_raw.items():
            try:
                target_layer_str, target_head_str = key.split("_")
                target_layer = int(target_layer_str)
                target_head = int(target_head_str)
                
                if not (isinstance(value.get("target_heads"), list) and len(value["target_heads"]) == 1 and
                        isinstance(value["target_heads"][0], list) and len(value["target_heads"][0]) == 2 and
                        value["target_heads"][0] == [target_layer, target_head]):
                    warnings.warn(f"Layer {layer_idx}: Skipping entry '{key}' due to unexpected target_heads format: {value.get('target_heads')}.", UserWarning)
                    continue
                
                ref_heads_raw = value.get("ref_heads")
                if not isinstance(ref_heads_raw, list) or not all(isinstance(item, list) and len(item) == 2 for item in ref_heads_raw):
                    warnings.warn(f"Layer {layer_idx}: Skipping entry '{key}' due to unexpected ref_heads format: {ref_heads_raw}.", UserWarning)
                    continue
                
                ref_heads = [(int(r[0]), int(r[1])) for r in ref_heads_raw]
                loaded_dependencies[(target_layer, target_head)] = ref_heads
                if target_layer not in targets_by_layer_temp:
                    targets_by_layer_temp[target_layer] = set()
                targets_by_layer_temp[target_layer].add(target_head)
                
            except (ValueError, KeyError, TypeError, IndexError) as e:
                warnings.warn(f"Layer {layer_idx}: Error parsing entry '{key}': {e}. Skipping entry.", UserWarning)
                continue
        
        if loaded_dependencies:
            targets_per_layer = {layer: sorted(list(heads)) for layer, heads in targets_by_layer_temp.items()}
            return loaded_dependencies, targets_per_layer
        else:
            warnings.warn(f"Layer {layer_idx}: No valid dependency entries found in '{dependency_file}'.", UserWarning)
            return {}, {}
            
    except Exception as e:
        warnings.warn(f"Layer {layer_idx}: Error loading/parsing dependency file '{dependency_file}': {e}.", UserWarning)
        return {}, {}

def _load_prediction_models(model_weights_file, target_head_dependencies, layer_idx, device, original_dtype):
    """Load and prepare prediction models."""
    if not os.path.exists(model_weights_file):
        warnings.warn(f"Layer {layer_idx}: Model weights file '{model_weights_file}' not found.", UserWarning)
        return {}
    
    try:
        with open(model_weights_file, "rb") as f:
            loaded_model_weights = pickle.load(f)
        
        if not isinstance(loaded_model_weights, dict):
            warnings.warn(f"Layer {layer_idx}: Model weights file '{model_weights_file}' not a dict.", UserWarning)
            return {}
        
        models_temp = {}
        models_prepared_count = 0
        
        for (target_layer, target_head), _ in target_head_dependencies.items():
            model_package = loaded_model_weights.get((target_layer, target_head))
            if not isinstance(model_package, dict):
                warnings.warn(f"Layer {layer_idx}, Target ({target_layer}, {target_head}): Invalid/missing model package.", UserWarning, stacklevel=2)
                continue
            
            # Support nn.Module or weight/bias payload
            if "model" in model_package and isinstance(model_package["model"], nn.Module):
                predictor_model = model_package["model"]
                predictor_model.to(device).eval().to(dtype=original_dtype)
                model_package["model"] = predictor_model
            elif "weights" in model_package and "bias" in model_package:
                # Construct linear layer from saved weights and bias
                W_saved, b_saved = model_package["weights"], model_package["bias"]
                
                # Convert numpy to tensor if needed
                W_t = torch.from_numpy(W_saved) if not isinstance(W_saved, torch.Tensor) else W_saved
                b_t = torch.from_numpy(b_saved) if not isinstance(b_saved, torch.Tensor) else b_saved
                
                # Move to correct device/dtype
                W_t = W_t.to(device=device, dtype=original_dtype)
                b_t = b_t.to(device=device, dtype=original_dtype)
                
                in_feats, out_feats = W_t.shape
                linear = nn.Linear(in_feats, out_feats, bias=True).to(device).to(dtype=original_dtype)
                linear.weight.data.copy_(W_t.t())
                linear.bias.data.copy_(b_t)
                linear.eval()
                
                model_package["model"] = linear
            else:
                warnings.warn(f"Layer {layer_idx}, Target ({target_layer}, {target_head}): No valid 'model' or weights in package.", UserWarning, stacklevel=2)
                continue
            
            models_temp[(target_layer, target_head)] = model_package
            models_prepared_count += 1
        
        if models_prepared_count > 0:
            return models_temp
        else:
            warnings.warn(f"Layer {layer_idx}: No models prepared from '{model_weights_file}'.", UserWarning)
            return {}
            
    except Exception as e:
        warnings.warn(f"Layer {layer_idx}: Error loading/unpickling model weights file '{model_weights_file}': {e}.", UserWarning)
        return {}

def _initialize_prediction_attributes(self):
    """Initialize prediction attributes if not already done."""
    if not hasattr(self, "prediction_initialized"):
        self.prediction_initialized = False
        self.key_prediction_enabled = False
        self.value_prediction_enabled = False
        self.target_head_dependencies_keys = {}
        self.targets_per_layer_keys = {}
        self.prediction_models_by_target_keys = {}
        self.target_head_dependencies_values = {}
        self.targets_per_layer_values = {}
        self.prediction_models_by_target_values = {}

def _lazy_load_prediction_data(self, device, original_dtype):
    """Lazy load prediction data once per instance."""
    if self.prediction_initialized:
        return
    
    self.prediction_initialized = True
    
    try:
        file_paths = _get_prediction_file_paths()
        
        # Load key dependencies and models
        self.target_head_dependencies_keys, self.targets_per_layer_keys = _load_dependencies(
            file_paths['dependency_file_keys'], self.layer_idx)
        can_load_key_dependencies = bool(self.target_head_dependencies_keys)
        
        self.prediction_models_by_target_keys = _load_prediction_models(
            file_paths['model_weights_file_keys'], self.target_head_dependencies_keys, 
            self.layer_idx, device, original_dtype) if can_load_key_dependencies else {}
        can_load_key_models = bool(self.prediction_models_by_target_keys)
        
        # Load value dependencies and models
        self.target_head_dependencies_values, self.targets_per_layer_values = _load_dependencies(
            file_paths['dependency_file_values'], self.layer_idx)
        can_load_value_dependencies = bool(self.target_head_dependencies_values)
        
        self.prediction_models_by_target_values = _load_prediction_models(
            file_paths['model_weights_file_values'], self.target_head_dependencies_values, 
            self.layer_idx, device, original_dtype) if can_load_value_dependencies else {}
        can_load_value_models = bool(self.prediction_models_by_target_values)
        
        # Enable prediction only if both dependencies and models loaded successfully
        self.key_prediction_enabled = can_load_key_dependencies and can_load_key_models
        self.value_prediction_enabled = can_load_value_dependencies and can_load_value_models
        
    except Exception as e:
        warnings.warn(f"Layer {self.layer_idx}: Unhandled error during lazy loading of prediction data: {e}. All prediction disabled.", UserWarning, stacklevel=2)
        self.key_prediction_enabled = False
        self.value_prediction_enabled = False
        self.target_head_dependencies_keys, self.targets_per_layer_keys, self.prediction_models_by_target_keys = {}, {}, {}
        self.target_head_dependencies_values, self.targets_per_layer_values, self.prediction_models_by_target_values = {}, {}, {}
    
    # Status print
    key_status = "DISABLED"
    if self.key_prediction_enabled:
        ready_targets = len(self.prediction_models_by_target_keys)
        total_targets = len(self.target_head_dependencies_keys)
        key_status = f"ENABLED for {ready_targets}/{total_targets} targets"
        
    value_status = "DISABLED"
    if self.value_prediction_enabled:
        ready_targets = len(self.prediction_models_by_target_values)
        total_targets = len(self.target_head_dependencies_values)
        value_status = f"ENABLED for {ready_targets}/{total_targets} targets"
    
    print(f"Layer {self.layer_idx}: Prediction initialized. Keys: {key_status} (requires Calibrate.predict_keys=True). Values: {value_status} (requires Calibrate.predict_values=True).")

def _gather_reference_states(ref_head_list, layer_idx, original_states, past_key_value, cache_type, q_len, cache_position, num_key_value_heads):
    """Gather reference states for prediction. Returns (collected_states, success)."""
    collected_ref_states = []
    
    for ref_layer, ref_head in ref_head_list:
        try:
            if not (0 <= ref_head < num_key_value_heads):
                warnings.warn(f"Layer {layer_idx}: Invalid ref head index {ref_head}. Skipping.", UserWarning, stacklevel=2)
                return [], False
            
            if ref_layer == layer_idx:
                ref_state = original_states[:, ref_head, :, :]
                collected_ref_states.append(ref_state)
            elif ref_layer < layer_idx:
                # Qwen3 specific cache access
                cache_attr = f"{cache_type}_cache"
                valid_cache = (past_key_value is not None and hasattr(past_key_value, cache_attr) and
                              ref_layer < len(getattr(past_key_value, cache_attr)) and
                              0 <= ref_layer < getattr(past_key_value, 'config', type('', (), {'num_hidden_layers': float('inf')})).num_hidden_layers and
                              past_key_value.get_seq_length(layer_idx=ref_layer) >= q_len)
                
                if not valid_cache:
                    if layer_idx > 0:
                        warnings.warn(f"Layer {layer_idx}: Cannot get ref ({ref_layer}, {ref_head}) {cache_type} cache for len {q_len}. Skipping.", UserWarning, stacklevel=1)
                    return [], False
                
                cache = getattr(past_key_value, cache_attr)[ref_layer]
                if q_len == 1:
                    current_token_index = cache_position[-1]
                    ref_state = cache[:, ref_head, current_token_index, :].unsqueeze(1)
                else:
                    ref_state = cache[:, ref_head, :q_len, :]
                collected_ref_states.append(ref_state)
            else:
                warnings.warn(f"Layer {layer_idx}: Ref layer {ref_layer} > current {layer_idx}. Skipping.", UserWarning, stacklevel=2)
                return [], False
                
        except Exception as e:
            warnings.warn(f"Layer {layer_idx}: Error gathering ref ({ref_layer}, {ref_head}): {e}. Skipping.", UserWarning, stacklevel=2)
            return [], False
    
    return collected_ref_states, bool(collected_ref_states)

def _perform_prediction(target_head_id, layer_idx, dependencies, models, original_states, past_key_value, 
                       cache_type, q_len, cache_position, num_key_value_heads, original_dtype, states_to_modify):
    """Perform prediction for a single target head and modify states in-place."""
    target_key = (layer_idx, target_head_id)
    
    if target_key not in models or target_key not in dependencies:
        return
    
    model_package = models[target_key]
    predictor_model = model_package["model"]
    ref_head_list = dependencies[target_key]
    
    # Gather reference states
    collected_ref_states, success = _gather_reference_states(
        ref_head_list, layer_idx, original_states, past_key_value, cache_type, 
        q_len, cache_position, num_key_value_heads)
    
    if not success:
        return
    
    try:
        # Prepare input & predict
        ref_input_stacked = torch.stack(collected_ref_states, dim=0)
        ref_input_permuted = ref_input_stacked.permute(1, 2, 0, 3)
        bsz_pred, _, num_refs, head_dim_pred = ref_input_permuted.shape
        predictor_input = ref_input_permuted.reshape(bsz_pred * q_len, -1)
        
        with torch.no_grad():
            predicted_state_flat = predictor_model(predictor_input).to(original_dtype)
        predicted_state = predicted_state_flat.view(bsz_pred, q_len, head_dim_pred).unsqueeze(1)
        
        # Replace state
        if q_len > 1:
            if Calibrate.apply_chat_template:
                states_to_modify[:, target_head_id:target_head_id+1, 30:-5, :] = predicted_state[:, :, 30:-5, :]
            else:
                states_to_modify[:, target_head_id:target_head_id+1, 2:, :] = predicted_state[:, :, 2:, :]
        elif q_len == 1:
            states_to_modify[:, target_head_id:target_head_id+1, :, :] = predicted_state
            
    except Exception as e:
        warnings.warn(f"Layer {layer_idx}, Target {target_key}: Error during prediction/replacement: {e}.", UserWarning, stacklevel=2)


def qwen3_atten_aug_forward_predict(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    device = key_states.device
    original_dtype = key_states.dtype

    # Initialize and lazy load prediction data
    _initialize_prediction_attributes(self)
    _lazy_load_prediction_data(self, device, original_dtype)

    # Determine if prediction should run
    predict_keys_now = self.key_prediction_enabled and Calibrate.predict_keys
    predict_values_now = self.value_prediction_enabled and Calibrate.predict_values
    if Calibrate.only_decode and q_len > 1:
        predict_keys_now = predict_values_now = False

    # Store original states if prediction might happen
    original_key_states = key_states.clone().detach() if predict_keys_now else None
    original_value_states = value_states.clone().detach() if predict_values_now else None

    # Perform Key Prediction
    if predict_keys_now and q_len > 0:
        current_layer_targets = self.targets_per_layer_keys.get(self.layer_idx, [])
        if not current_layer_targets:
            warnings.warn(f"Layer {self.layer_idx}: Key prediction enabled, but no target heads found for this layer. Skipping key prediction.", UserWarning, stacklevel=2)
        else:
            for target_head_id in current_layer_targets:
                _perform_prediction(target_head_id, self.layer_idx, self.target_head_dependencies_keys, 
                                   self.prediction_models_by_target_keys, original_key_states, past_key_value,
                                   'key', q_len, cache_position, self.config.num_key_value_heads, original_dtype, key_states)

    # Perform Value Prediction
    if predict_values_now and q_len > 0:
        current_layer_targets = self.targets_per_layer_values.get(self.layer_idx, [])
        if not current_layer_targets:
            warnings.warn(f"Layer {self.layer_idx}: Value prediction enabled, but no target heads found for this layer. Skipping value prediction.", UserWarning, stacklevel=2)
        else:
            for target_head_id in current_layer_targets:
                _perform_prediction(target_head_id, self.layer_idx, self.target_head_dependencies_values,
                                   self.prediction_models_by_target_values, original_value_states, past_key_value,
                                   'value', q_len, cache_position, self.config.num_key_value_heads, original_dtype, value_states)

    # KV Cache Update
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
        sliding_window=self.sliding_window,  # diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights