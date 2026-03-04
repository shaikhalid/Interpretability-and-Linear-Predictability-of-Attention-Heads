import torch
from typing import Dict, List, Tuple, Set

class SharedKVCacheManager:
    """
    Manages shared KV cache between selected attention heads across different layers.
    """
    def __init__(self):
        self.sharing_groups = []  # List of lists of (layer, head) tuples that share cache
        self.enabled = False
        self.kv_cache = {}  # Dictionary to store shared KV states

    def set_sharing_groups(self, groups: List[List[Tuple[int, int]]]):
        """
        Set the groups of heads that should share KV cache.
        
        Args:
            groups: List of lists, where each inner list contains (layer_idx, head_idx) tuples
                   that should share the same KV cache.
        """
        self.sharing_groups = groups
        self.enabled = len(groups) > 0
        self.kv_cache = {}
        
        # Create a lookup map for faster access
        self.sharing_map = {}
        for group_idx, group in enumerate(groups):
            for layer_idx, head_idx in group:
                self.sharing_map[(layer_idx, head_idx)] = group_idx
    
    def get_group_id(self, layer_idx: int, head_idx: int) -> int:
        """
        Returns the sharing group ID for a given layer and head index.
        Returns -1 if the head does not belong to any sharing group.
        """
        return self.sharing_map.get((layer_idx, head_idx), -1)
    
    def is_in_sharing_group(self, layer_idx: int, head_idx: int) -> bool:
        """Check if a given layer/head is part of any sharing group."""
        return (layer_idx, head_idx) in self.sharing_map
    
    def get_sharing_group(self, layer_idx: int, head_idx: int) -> List[Tuple[int, int]]:
        """Get all layer/head pairs that share cache with the given layer/head."""
        group_id = self.get_group_id(layer_idx, head_idx)
        if group_id == -1:
            return []
        return self.sharing_groups[group_id]
    
    def store_kv_states(self, layer_idx: int, head_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        """Store KV states for a sharing group if the layer/head belongs to one."""
        group_id = self.get_group_id(layer_idx, head_idx)
        if group_id != -1:
            # Validate shapes
            if key_states.size(1) == 0 or value_states.size(1) == 0:
                print(f"WARNING: Attempting to store empty tensor for group {group_id}, layer {layer_idx}, head {head_idx}")
                print(f"Key shape: {key_states.shape}, Value shape: {value_states.shape}")
                return
            
            self.kv_cache[group_id] = (key_states.clone(), value_states.clone())
    
    def get_kv_states(self, layer_idx: int, head_idx: int):
        """Get KV states for a layer/head if it belongs to a sharing group."""
        group_id = self.get_group_id(layer_idx, head_idx)
        if group_id != -1 and group_id in self.kv_cache:
            return self.kv_cache[group_id]
        return None

# Create a global instance that can be accessed from anywhere
shared_kv_manager = SharedKVCacheManager() 