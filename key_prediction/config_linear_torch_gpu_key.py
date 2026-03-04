# Configuration for Linear Regression PyTorch GPU Model

CONFIG = {
    # --- Data & Feature Settings ---
    'file_path': 'kv_pickles/k_cache_data_truthfulqa_mc1_l8b.pkl',  # Or your k_cache data file
    # 'num_layers': 32, # Removed
    'layer_start': 0,   # NEW: Starting layer index (inclusive)
    'layer_end': 70,    # NEW: Ending layer index (inclusive)
    # 'ref_heads': [0, 1], # Ignored if dynamic_ref_heads is True
    # 'target_heads': [2, 3, 4, 5, 6, 7], # Ignored if dynamic_ref_heads is True
    'matrix_type': 'k_matrix',  # Change to 'k_matrix' if using k_cache data
    #REMEMBER TO SET THIS TO NONE for HUMANEVAL non instruction 
    'keep_instruction_samples': None, # NEW: Keep instruction tokens for the first N samples per layer/head. Set to 0 to remove for all.
    'include_prefill': True, # NEW: Set to False to exclude prefill tokens from training/evaluation data

    # --- Dynamic Head Selection (NEW - Layer-wise Reference Selection) ---
    'dynamic_ref_heads': True,      # Set to True to enable dynamic selection
    # 'ref_head_metric': 'cos_sim_mean', # Metric to rank heads based on their avg performance *as a reference*
                                      # predicting all other heads. ('cos_sim_mean', 'mse', 'r2')
                                      # Higher is better for cos_sim_mean and r2, lower for mse.
    # 'target_selection_metric': 'cos_sim_mean', # NO LONGER USED when dynamic_ref_heads is True
    'save_head_selection': False,     # NEW: Save selected ref/target heads to JSON after dynamic search
    'load_head_selection': True,    # NEW: Load head selections from JSON instead of searching/using static
    'head_selection_file': 'ref_target_mapping/ref_target_heads_keys.json', # NEW: File for saving/loading head selections

    # --- Model Settings ---
    'model_type': 'linear_torch_gpu', # <--- Specify the PyTorch GPU linear model
    'precision': 'float32', 
    'save_model_weights': True, # NEW: Set to True to save trained model weights
    'model_weights_file': 'model_pickles/all_model_weights_keys.pkl', # NEW: Filename for single weights file

    # --- Training Settings ---
    'test_size': 0.25,
    'random_state': 42,
    'learning_rate': 0.1,
    'epochs': 250,
    'batch_size': 256,
    'regularization': 0,
    # Add relevant model hyperparameters here if needed (e.g., alpha for Ridge)
    # 'alpha': 1.0, # Example for Ridge
    # 'alpha_search': False, # Example: Set to True to enable alpha search for Ridge

    # --- Evaluation Settings ---
    'low_similarity_threshold': 0.85, # Only used if matrix_type is 'k_matrix'

    # --- Add other relevant parameters here if needed ---
    'save_csv_results': True, # NEW: Set to False to disable saving CSV result files
} 