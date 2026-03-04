# ============================================================================ #
#  OLMo‑2 attention with optional Key/Value prediction                         #
#  – requires the same Calibrate flags, JSON mappings, and PKL regressors      #
#    that the Llama patch uses.                                                #
#  – call it instead of models.modelling_olmo2.Attention.forward               #
# ============================================================================ #
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import warnings
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from torch import nn
from transformers.cache_utils import Cache
from models.modelling_olmo2 import eager_attention_forward, apply_rotary_pos_emb
from models.calibrate import Calibrate

def olmo2_atten_aug_forward_predict(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Forward pass identical to vanilla OLMo attention **except** that, when
    enabled, it predicts selected K/V heads from a small set of reference heads
    using pretrained regressors, then overwrites those heads before computing
    attention.
    """
    # ------------------------------------------------------------------ #
    # 1. Standard Q / K / V projection + RoPE                            #
    # ------------------------------------------------------------------ #
    bsz, q_len, _ = hidden_states.size()
    hidden_shape = (bsz, q_len, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states))
    key_states   = self.k_norm(self.k_proj(hidden_states))
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(hidden_shape).transpose(1, 2)      # (B, H, T, Dh)
    key_states   = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # ------------------------------------------------------------------ #
    # 2. One‑time lazy initialisation of prediction machinery            #
    # ------------------------------------------------------------------ #
    if not hasattr(self, "prediction_initialized"):
        self.prediction_initialized       = False
        self.key_prediction_enabled       = False
        self.value_prediction_enabled     = False

        # slot for mappings / models
        self.target_head_dependencies_keys   = {}
        self.targets_per_layer_keys          = {}
        self.prediction_models_by_target_keys = {}
        self.target_head_dependencies_values = {}
        self.targets_per_layer_values        = {}
        self.prediction_models_by_target_values = {}

    if not self.prediction_initialized:
        self.prediction_initialized = True     # make sure we run this only once
        device, dtype = key_states.device, key_states.dtype

        def _lazy_load(kind: str):
            """kind ∈ {'keys', 'values'} – returns (deps_ok, models_ok)."""
            dep_file = f"ref_target_mapping/ref_target_heads_{Calibrate.dataset}_{Calibrate.model_alias}_{kind}.json"
            mdl_file = f"model_pickles/all_model_weights_{Calibrate.dataset}_{Calibrate.model_alias}_{kind}.pkl"

            deps_ok, models_ok = False, False

            if os.path.exists(dep_file):
                try:
                    import json, os
                    with open(dep_file, "r") as f:
                        raw = json.load(f)
                    deps_tmp, layer_map = {}, {}
                    for k, v in raw.items():
                        try:
                            tl, th = map(int, k.split("_"))
                            ref = [(int(r[0]), int(r[1])) for r in v["ref_heads"]]
                            deps_tmp[(tl, th)] = ref
                            layer_map.setdefault(tl, []).append(th)
                        except Exception as e:
                            warnings.warn(f"Layer {self.layer_idx}: bad entry {k} in {dep_file}: {e}")
                    if deps_tmp:
                        if kind == "keys":
                            self.target_head_dependencies_keys = deps_tmp
                            self.targets_per_layer_keys = {l: sorted(hs) for l, hs in layer_map.items()}
                        else:
                            self.target_head_dependencies_values = deps_tmp
                            self.targets_per_layer_values = {l: sorted(hs) for l, hs in layer_map.items()}
                        deps_ok = True
                except Exception as e:
                    warnings.warn(f"Could not load {dep_file}: {e}")

            if os.path.exists(mdl_file):
                try:
                    import pickle
                    with open(mdl_file, "rb") as f:
                        loaded = pickle.load(f)
                    ready = {}
                    for tgt, pkg in loaded.items():
                        model = pkg.get("model")
                        if isinstance(model, nn.Module):
                            model.to(device).eval().to(dtype=dtype)
                            ready[tgt] = pkg
                    if ready:
                        if kind == "keys":
                            self.prediction_models_by_target_keys = ready
                        else:
                            self.prediction_models_by_target_values = ready
                        models_ok = True
                except Exception as e:
                    warnings.warn(f"Could not load {mdl_file}: {e}")

            return deps_ok, models_ok

        k_dep, k_mdl = _lazy_load("keys")
        v_dep, v_mdl = _lazy_load("values")
        self.key_prediction_enabled   = k_dep and k_mdl
        self.value_prediction_enabled = v_dep and v_mdl

        print(
            f"Layer {self.layer_idx}: Prediction initialised. "
            f"Keys: {'ENABLED' if self.key_prediction_enabled else 'DISABLED'} | "
            f"Values: {'ENABLED' if self.value_prediction_enabled else 'DISABLED'}"
        )

    # ------------------------------------------------------------------ #
    # 3. Decide whether to predict on this forward call                  #
    # ------------------------------------------------------------------ #
    predict_keys_now   = self.key_prediction_enabled   and Calibrate.predict_keys
    predict_values_now = self.value_prediction_enabled and Calibrate.predict_values

    if Calibrate.only_decode and q_len > 1:
        predict_keys_now = predict_values_now = False

    # Keep originals for reference‑head gathering
    original_key_states   = key_states.clone()   if predict_keys_now   else None
    original_value_states = value_states.clone() if predict_values_now else None

    num_heads = key_states.size(1)  # works for K & V

    # ------------------------------------------------------------------ #
    # 4a. Key‑state prediction                                           #
    # ------------------------------------------------------------------ #
    if predict_keys_now and q_len > 0:
        cur_targets = self.targets_per_layer_keys.get(self.layer_idx, [])
        for tgt_head in cur_targets:
            tgt = (self.layer_idx, tgt_head)
            if tgt not in self.prediction_models_by_target_keys:
                continue

            ref_list  = self.target_head_dependencies_keys[tgt]
            refs_ok, ref_vecs = True, []
            for rl, rh in ref_list:
                try:
                    if not (0 <= rh < num_heads):
                        refs_ok = False
                        break
                    if rl == self.layer_idx:
                        ref_vecs.append(original_key_states[:, rh, :, :])
                    elif rl < self.layer_idx and past_key_value is not None:
                        if q_len == 1:
                            tok = cache_position[-1]
                            vec = past_key_value.key_cache[rl][:, rh, tok, :].unsqueeze(1)
                        else:
                            vec = past_key_value.key_cache[rl][:, rh, :q_len, :]
                        ref_vecs.append(vec)
                    else:
                        refs_ok = False
                        break
                except Exception:
                    refs_ok = False
                    break
            if not refs_ok or not ref_vecs:
                continue

            ref_stacked   = torch.stack(ref_vecs, 0)                      # (R, B, T, Dh)
            ref_perm      = ref_stacked.permute(1, 2, 0, 3)               # (B, T, R, Dh)
            B, T, R, Dh   = ref_perm.shape
            predictor_in  = ref_perm.reshape(B * T, R * Dh)

            with torch.no_grad():
                pred_flat = self.prediction_models_by_target_keys[tgt]["model"](predictor_in).to(key_states.dtype)
            pred_vec = pred_flat.view(B, 1, T, Dh)

            key_states[:, tgt_head:tgt_head + 1, :, :] = pred_vec

    # ------------------------------------------------------------------ #
    # 4b. Value‑state prediction                                         #
    # ------------------------------------------------------------------ #
    if predict_values_now and q_len > 0:
        cur_targets = self.targets_per_layer_values.get(self.layer_idx, [])
        for tgt_head in cur_targets:
            tgt = (self.layer_idx, tgt_head)
            if tgt not in self.prediction_models_by_target_values:
                continue

            ref_list = self.target_head_dependencies_values[tgt]
            refs_ok, ref_vecs = True, []
            for rl, rh in ref_list:
                try:
                    if not (0 <= rh < num_heads):
                        refs_ok = False
                        break
                    if rl == self.layer_idx:
                        ref_vecs.append(original_value_states[:, rh, :, :])
                    elif rl < self.layer_idx and past_key_value is not None:
                        if q_len == 1:
                            tok = cache_position[-1]
                            vec = past_key_value.value_cache[rl][:, rh, tok, :].unsqueeze(1)
                        else:
                            vec = past_key_value.value_cache[rl][:, rh, :q_len, :]
                        ref_vecs.append(vec)
                    else:
                        refs_ok = False
                        break
                except Exception:
                    refs_ok = False
                    break
            if not refs_ok or not ref_vecs:
                continue

            ref_stacked  = torch.stack(ref_vecs, 0)
            ref_perm     = ref_stacked.permute(1, 2, 0, 3)
            B, T, R, Dh  = ref_perm.shape
            predictor_in = ref_perm.reshape(B * T, R * Dh)

            with torch.no_grad():
                pred_flat = self.prediction_models_by_target_values[tgt]["model"](predictor_in).to(value_states.dtype)
            pred_vec = pred_flat.view(B, 1, T, Dh)

            value_states[:, tgt_head:tgt_head + 1, :, :] = pred_vec

    # ------------------------------------------------------------------ #
    # 5. Update KV cache **after** any overwrites                        #
    # ------------------------------------------------------------------ #
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # ------------------------------------------------------------------ #
    # 6. Run attention (same as vanilla path)                            #
    # ------------------------------------------------------------------ #
    attention_fn = (
        eager_attention_forward
        if self.config._attn_implementation == "eager"
        or (self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False))
        else ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    )

    attn_output, attn_weights = attention_fn(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # ------------------------------------------------------------------ #
    # 7. Final projection                                                #
    # ------------------------------------------------------------------ #
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
