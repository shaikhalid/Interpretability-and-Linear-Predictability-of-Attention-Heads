# Configuration for Linear Regression GPU Model

CONFIG = {
    # --- Data & Feature Settings ---
    'file_path': 'k_cache_data_humaneval_32b.pkl',  # Or your k_cache data file
    'num_layers': 32,
    # 'ref_heads': [0, 1], # Ignored if dynamic_ref_heads is True
    # 'target_heads': [2, 3, 4, 5, 6, 7], # Ignored if dynamic_ref_heads is True
    'matrix_type': 'k_matrix',  # Change to 'k_matrix' if using k_cache data

    # --- Dynamic Head Selection (NEW - Layer-wise Reference Selection) ---
    'dynamic_ref_heads': True,      # Set to True to enable dynamic selection
    'ref_head_metric': 'cos_sim_mean', # Metric to rank heads based on their avg performance *as a reference*
                                      # predicting all other heads. ('cos_sim_mean', 'mse', 'r2')
                                      # Higher is better for cos_sim_mean and r2, lower for mse.
    # 'target_selection_metric': 'cos_sim_mean', # NO LONGER USED when dynamic_ref_heads is True
    'save_head_selection': True,     # NEW: Save selected ref/target heads to JSON after dynamic search
    'load_head_selection': False,    # NEW: Load head selections from JSON instead of searching/using static
    'head_selection_file': 'selected_heads.json', # NEW: File for saving/loading head selections

    # --- Model Settings ---
    'model_type': 'linear_gpu', # <--- Specify the GPU linear model
    'precision': 'float64', # Optional: Use 'float32' for potentially faster GPU training, defaults to 'float64'
    'save_model_weights': True, # NEW: Set to True to save trained model weights
    'model_weights_file': 'all_model_weights.pkl', # NEW: Filename for single weights file

    # --- Training Settings ---
    'test_size': 0.25,
    'random_state': 42,
    # Add relevant model hyperparameters here if needed (e.g., alpha for Ridge)
    # 'alpha': 1.0, # Example for Ridge
    # 'alpha_search': False, # Example: Set to True to enable alpha search for Ridge

    # --- Evaluation Settings ---
    'low_similarity_threshold': 0.85, # Only used if matrix_type is 'k_matrix'

    # --- Add other relevant parameters here if needed ---
    'save_csv_results': False, # NEW: Set to False to disable saving CSV result files
}