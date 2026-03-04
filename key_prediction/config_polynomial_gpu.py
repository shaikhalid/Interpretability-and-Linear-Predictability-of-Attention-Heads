# Configuration for Polynomial Regression on GPU Model

CONFIG = {
    'file_path': 'v_cache_data_humaneval.pkl',
    'num_layers': 32,
    'ref_heads': [0, 2, 4, 6],
    'target_heads': [1, 3, 5, 7],
    'test_size': 0.25,
    'random_state': 42,
    'low_similarity_threshold': 0.85,
    'model_type': 'polynomial_gpu', # 'linear', 'polynomial_cpu', or 'polynomial_gpu'
    # Polynomial Regression specific config
    'poly_degree': 2,
    # PCA specific config
    'pca1_n_components': 200,  # Or 128, 64, etc. - MUST be an integer < 512
    'pca2_n_components': 0.97,
    'poly_interaction_only': True, # Whether to include only interaction features
} 