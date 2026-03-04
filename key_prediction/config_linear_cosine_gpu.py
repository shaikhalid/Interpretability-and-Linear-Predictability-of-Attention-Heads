# Configuration for Custom Cosine Similarity Linear Regression GPU Model

CONFIG = {
    # --- Data & Feature Settings ---
    'file_path': 'k_cache_data_humaneval.pkl',  # Or your k_cache data file
    'num_layers': 32,
    'ref_heads': [0, 2, 4, 6],
    'target_heads': [1, 3, 5, 7],
    'matrix_type': 'k_matrix',  # Change to 'k_matrix' if using k_cache data

    # --- Model Settings ---
    'model_type': 'linear_cosine_gpu', # <--- Specify the custom GPU cosine model
    'precision': 'float32', # Use 'float32' for potentially faster GPU training, defaults to 'float64'

    # --- Custom Optimizer Settings ---
    'learning_rate': 0.5,
    'epochs': 100,
    'regularization': 0.0001, # Optional L2 regularization strength
    'batch_size': 128, # Optional batch size for mini-batch gradient descent

    # --- Training Settings ---
    'test_size': 0.25,
    'random_state': 42,

    # --- Evaluation Settings ---
    'low_similarity_threshold': 0.85, # Only used if matrix_type is 'k_matrix'

    # --- Add other relevant parameters here if needed ---
} 