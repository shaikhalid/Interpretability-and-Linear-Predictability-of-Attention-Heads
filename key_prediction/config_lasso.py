# Configuration for Lasso Regression Model

CONFIG = {
    'file_path': 'k_cache_data_humaneval.pkl',
    'num_layers': 32,
    'ref_heads': [0, 2, 4, 6],
    'target_heads': [1, 3, 5, 7],
    'test_size': 0.25,
    'random_state': 42,
    'low_similarity_threshold': 0.85,
    'model_type': 'lasso',
    'alpha': 1.0,  # Regularization strength for Lasso
    # Add other relevant parameters here if needed
} 