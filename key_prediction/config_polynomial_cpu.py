# Configuration for Polynomial Regression with PCA Model (CPU)

CONFIG = {
    'file_path': 'v_cache_data_humaneval.pkl',
    'matrix_type': 'v_matrix',
    'num_layers': 32,
    'ref_heads': [0, 2, 4, 6],
    'target_heads': [1, 3, 5, 7],
    'test_size': 0.25,
    'random_state': 42,
    'low_similarity_threshold': 0.85,
    'model_type': 'polynomial_cpu', # 'linear' or 'polynomial_cpu'
    # Polynomial Regression with PCA specific config
    'pca_n_components': 0.95, # Keep 95% variance, or set integer for num components
    'poly_degree': 2,
} 