# Configuration for Ridge Regression Model

CONFIG = {
    'file_path': 'v_cache_data_humaneval.pkl',
    'num_layers': 32,
    'ref_heads': [0, 2, 4, 6],
    'target_heads': [1, 3, 5, 7],
    'test_size': 0.25,
    'random_state': 42,
    'low_similarity_threshold': 0.85,
    'model_type': 'ridge',
    'matrix_type': 'v_matrix',  # Use v_matrix instead of k_matrix
    'alpha': 1.0,  # Regularization strength for Ridge
    'alpha_search': True,  # Enable alpha parameter search
    'alpha_values': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],  # Alpha values to search
    'cv_folds': 5,  # Number of cross-validation folds for alpha search
    # Add other relevant parameters here if needed
} 