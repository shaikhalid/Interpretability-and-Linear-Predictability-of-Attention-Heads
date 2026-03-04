# Key Prediction Models

This directory contains code for training and evaluating different regression models to predict attention key vectors in a transformer model based on the key vectors from reference heads in the same layer.

## Files

*   `trainer.py`: The main script to load data, preprocess it, train a specified model, evaluate its performance (R2, MSE, Cosine Similarity), and analyze results.
*   `models.py`: Defines the different regression models (Linear, Polynomial, Lasso, Ridge) and their training/evaluation logic, supporting both CPU (scikit-learn) and GPU (cuML).
*   `config_*.py`: Configuration files for different model types. These files specify parameters like data path, layer/head configuration, model hyperparameters, etc.

## Available Models & Configurations

The `trainer.py` script selects the model based on the `model_type` specified in the configuration file passed via the `--config` argument.

Available `model_type` values:

*   `linear`: Standard Linear Regression (CPU).
    *   Config: `config_linear.py`
*   `polynomial_cpu`: Polynomial Regression with PCA preprocessing (CPU).
    *   Config: `config_polynomial_cpu.py`
    *   Hyperparameters: `pca1_n_components`, `pca2_n_components`, `poly_degree`.
*   `polynomial_gpu`: Polynomial Regression with PCA preprocessing (GPU via cuML).
    *   Config: `config_polynomial_gpu.py`
    *   Hyperparameters: `pca1_n_components`, `pca2_n_components`, `poly_degree`, `precision` (`float32` or `float64`).
*   `lasso`: Lasso Regression (CPU).
    *   Config: `config_lasso.py`
    *   Hyperparameters: `alpha` (regularization strength).
*   `lasso_gpu`: Lasso Regression (GPU via cuML).
    *   Config: Create a config file (e.g., `config_lasso_gpu.py`) and set `model_type: 'lasso_gpu'`.
    *   Hyperparameters: `alpha`, `precision`.
*   `ridge`: Ridge Regression (CPU).
    *   Config: `config_ridge.py`
    *   Hyperparameters: `alpha`.
*   `ridge_gpu`: Ridge Regression (GPU via cuML).
    *   Config: Create a config file (e.g., `config_ridge_gpu.py`) and set `model_type: 'ridge_gpu'`.
    *   Hyperparameters: `alpha`, `precision`.

## How to Run

1.  **Ensure Dependencies**: Make sure you have the necessary libraries installed (pandas, numpy, scikit-learn, matplotlib, seaborn). For GPU models, you need a CUDA-enabled GPU and cuML/CuPy installed.
2.  **Prepare Data**: Ensure the data file specified in the `file_path` of your chosen config file (e.g., `k_cache_data_humaneval.pkl`) exists and is accessible.
3.  **Choose Configuration**: Select or create a configuration file (`config_*.py`) corresponding to the model you want to run.
4.  **Run Trainer**: Execute the `trainer.py` script from the parent directory (e.g., `C-SAF-copy1`), passing the path to your chosen configuration file using the `--config` argument:

    ```bash
    # Example for Linear Regression (CPU)
    python -m key_prediction.trainer --config key_prediction/config_linear.py

    # Example for Lasso Regression (CPU)
    python -m key_prediction.trainer --config key_prediction/config_lasso.py

    # Example for Ridge Regression (CPU) - assuming config_ridge_gpu.py exists
    python -m key_prediction.trainer --config key_prediction/config_ridge.py

    # Example for Linear Regression (GPU)
    python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu.py

    python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py
    
    # Example for Polynomial Regression (GPU)
    python -m key_prediction.trainer --config key_prediction/config_polynomial_gpu.py

    ```

## Output

The script will print performance metrics (R2, MSE, Cosine Similarity) for each target head in each layer. It will also print summaries and analysis of low-similarity predictions. If run from the command line, it saves the following files to the directory where you run the command:

*   `model_results.csv`: Detailed results for each layer/head combination.
*   `layer_summary.csv`: Average results per layer.
*   `low_similarity_samples.csv`: Details of individual predictions below the similarity threshold.
*   `low_similarity_summary.csv`: Summary statistics of low-similarity predictions.

Plots visualizing the results might be displayed if running in an environment that supports GUI rendering (like a local machine with Matplotlib backend configured). They are not saved automatically by default. 