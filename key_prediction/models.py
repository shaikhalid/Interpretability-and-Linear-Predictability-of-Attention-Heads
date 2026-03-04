import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, Normalizer as SklearnNormalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# Add cuML imports
import cupy as cp
from cuml.linear_model import LinearRegression as cuLinearRegression, Lasso as cuLasso, Ridge as cuRidge
from cuml.preprocessing import PolynomialFeatures as cuPolynomialFeatures, Normalizer as cuMLNormalizer
from cuml.metrics import r2_score as cu_r2_score
from cuml.metrics import mean_squared_error as cu_mean_squared_error
from cuml.decomposition import PCA as cuPCA
import pandas as pd # Added for DataFrame check in evaluate_predictions
# Add PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_and_evaluate_model(X_train, X_test, y_train, y_test, test_sample_info=None, config=None):
    """
    Train a linear regression model and evaluate its performance.
    Performs CPU normalization if matrix_type is k_matrix.

    Args:
        X_train, X_test, y_train, y_test: Training and test data
        test_sample_info: DataFrame containing sample info for test data
        config: Dictionary containing configuration parameters

    Returns:
        r2_test, mse_test, cosine_similarities_test, low_sim_sample_info: Performance metrics and low similarity sample info
        model_package (dict): Dictionary containing the trained model and normalizers ('model', 'x_normalizer', 'y_normalizer', 'normalized_training').
    """
    # Get config parameters
    matrix_type = config.get('matrix_type', 'k_matrix') if config else 'k_matrix'
    low_sim_threshold = config.get('low_similarity_threshold', 0.85) if config else 0.85
    x_normalizer = None # Initialize normalizers
    y_normalizer = None

    # Normalize data on CPU if k_matrix
    if matrix_type == 'k_matrix':
        print("Normalizing data (k_matrix) on CPU")
        x_normalizer = SklearnNormalizer()
        X_train = x_normalizer.fit_transform(X_train)
        X_test = x_normalizer.transform(X_test)

        y_normalizer = SklearnNormalizer()
        y_train = y_normalizer.fit_transform(y_train)
        y_test = y_normalizer.transform(y_test) # Normalize y_test for evaluation consistency
    else:
        print("Skipping normalization (not k_matrix)")

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained on the training set.")

    # Predict on the Test Set
    y_pred_test = model.predict(X_test)
    print("Predictions generated for the test set.")

    # Calculate Cosine Similarity (Test Set)
    # Ensure y_test and y_pred_test are numpy arrays
    y_test_np = np.array(y_test)
    y_pred_test_np = np.array(y_pred_test)

    # Calculate metrics via evaluate_predictions
    r2_test, mse_test, cosine_similarities_test, low_sim_sample_info = evaluate_predictions(
        y_test_np, y_pred_test_np, test_sample_info, low_sim_threshold, use_gpu=False, matrix_type=matrix_type
    )

    model_package = {
        'model': model,
        'x_normalizer': x_normalizer,
        'y_normalizer': y_normalizer,
        'normalized_training': matrix_type == 'k_matrix'
    }

    return r2_test, mse_test, cosine_similarities_test, low_sim_sample_info, model_package

def train_and_evaluate_model_gpu(X_train, X_test, y_train, y_test, test_sample_info=None, config=None):
    """
    Train a Linear Regression model on the GPU and evaluate its performance.
    Performs GPU normalization if matrix_type is k_matrix.

    Args:
        X_train, X_test, y_train, y_test: Training and test data (numpy arrays)
        test_sample_info: DataFrame containing sample info for test data
        config: Dictionary containing configuration parameters

    Returns:
        r2_test, mse_test, cosine_similarities_test, low_sim_sample_info: Performance metrics and low similarity sample info
        model_package (dict): Dictionary containing the trained model and normalizers ('model', 'x_normalizer', 'y_normalizer', 'normalized_training').
    """
    # Get config parameters
    matrix_type = config.get('matrix_type', 'k_matrix') if config else 'k_matrix'
    low_sim_threshold = config.get('low_similarity_threshold', 0.85) if config else 0.85
    precision = config.get('precision', 'float64')
    x_normalizer = None # Initialize normalizers
    y_normalizer = None

    dtype = cp.float64 if precision == 'float64' else cp.float32
    print(f"Using {precision} precision for GPU computation.")

    # Convert data to CuPy arrays
    X_train_gpu = cp.asarray(X_train, dtype=dtype)
    y_train_gpu = cp.asarray(y_train, dtype=dtype)
    X_test_gpu = cp.asarray(X_test, dtype=dtype)
    y_test_gpu = cp.asarray(y_test, dtype=dtype)

    # Normalize data on GPU if k_matrix
    if matrix_type == 'k_matrix':
        print("Normalizing data (k_matrix) on GPU")
        x_normalizer = cuMLNormalizer()
        X_train_gpu = x_normalizer.fit_transform(X_train_gpu)
        X_test_gpu = x_normalizer.transform(X_test_gpu)

        # DO NOT NORMALIZE y_train_gpu or y_test_gpu
        # y_normalizer = cuMLNormalizer()
        # y_train_gpu = y_normalizer.fit_transform(y_train_gpu)
        # y_test_gpu = y_normalizer.transform(y_test_gpu) # Normalize y_test for evaluation consistency
    else:
        print("Skipping normalization (not k_matrix)")

    # Ensure y_train_gpu is C-contiguous for cuML
    if y_train_gpu.ndim > 1 and y_train_gpu.shape[1] > 1 and not y_train_gpu.flags['C_CONTIGUOUS']:
        y_train_gpu = cp.ascontiguousarray(y_train_gpu)

    # Train Linear Regression Model on GPU
    print("Training Linear Regression model on GPU...")
    model = cuLinearRegression(fit_intercept=True, algorithm='svd') # SVD is generally stable
    try:
        model.fit(X_train_gpu, y_train_gpu)
        print("Linear Regression model trained on GPU.")
        # Predict on the Test Set
        y_pred_test_gpu = model.predict(X_test_gpu)
        print("Predictions generated for the test set.")
    except Exception as e:
        print(f"Error during Linear Regression GPU fit/predict: {e}")
        # Return default bad values and None for model package
        return -1, float('inf'), cp.array([]), None, None

    # Evaluation
    print("Evaluating predictions...")
    r2_test, mse_test, cosine_similarities_test, low_sim_sample_info = evaluate_predictions(
        y_test_gpu, y_pred_test_gpu, test_sample_info, low_sim_threshold, use_gpu=True, matrix_type=matrix_type
    )

    model_package = {
        'model': model,
        'x_normalizer': x_normalizer,
        'y_normalizer': None, # Explicitly set y_normalizer to None
        'normalized_training': matrix_type == 'k_matrix' # Indicate X was normalized
    }

    return r2_test, mse_test, cosine_similarities_test, low_sim_sample_info, model_package

def train_and_evaluate_linear_cosine_gpu(X_train, X_test, y_train, y_test, test_sample_info=None, config=None):
    """
    Train a Linear Regression model on the GPU by directly optimizing for Cosine Similarity
    using gradient descent. Loss = 1 - CosineSimilarity.
    Performs GPU normalization if matrix_type is k_matrix.

    Args:
        X_train, X_test, y_train, y_test: Training and test data (numpy arrays)
        test_sample_info: DataFrame containing sample info for test data
        config: Dictionary containing configuration parameters (learning_rate, epochs, regularization, etc.)

    Returns:
        r2_test, mse_test, cosine_similarities_test, low_sim_sample_info: Performance metrics and low similarity sample info
        model_package (dict): Dictionary containing the trained model ('W', 'b') and normalizers ('x_normalizer', 'y_normalizer', 'normalized_training').
    """
    # Get config parameters
    matrix_type = config.get('matrix_type', 'k_matrix') if config else 'k_matrix'
    low_sim_threshold = config.get('low_similarity_threshold', 0.85) if config else 0.85
    precision = config.get('precision', 'float32') # Default to float32 for speed
    learning_rate = config.get('learning_rate', 0.01) if config else 0.01
    epochs = config.get('epochs', 100) if config else 100
    regularization = config.get('regularization', 0.0) if config else 0.0 # L2 regularization
    batch_size = config.get('batch_size', None) if config else None # Use full batch if None
    epsilon = 1e-9 # Small value for numerical stability
    x_normalizer = None # Initialize normalizers
    y_normalizer = None

    dtype = cp.float64 if precision == 'float64' else cp.float32
    print(f"Using {precision} precision for Custom Cosine LR GPU computation.")
    print(f"Optimizer settings: LR={learning_rate}, Epochs={epochs}, L2 Reg={regularization}, Batch Size={batch_size or 'Full'}")

    # Convert data to CuPy arrays
    X_train_gpu = cp.asarray(X_train, dtype=dtype)
    y_train_gpu = cp.asarray(y_train, dtype=dtype)
    X_test_gpu = cp.asarray(X_test, dtype=dtype)
    y_test_gpu = cp.asarray(y_test, dtype=dtype)

    n_samples, n_features = X_train_gpu.shape
    n_targets = y_train_gpu.shape[1]

    # Normalize data on GPU if k_matrix
    if matrix_type == 'k_matrix':
        print("Normalizing data (k_matrix) on GPU")
        x_normalizer = cuMLNormalizer()
        X_train_gpu = x_normalizer.fit_transform(X_train_gpu)
        X_test_gpu = x_normalizer.transform(X_test_gpu)

        y_normalizer = cuMLNormalizer()
        y_train_gpu = y_normalizer.fit_transform(y_train_gpu)
        y_test_gpu = y_normalizer.transform(y_test_gpu) # Normalize y_test for evaluation consistency
    else:
        print("Skipping normalization (not k_matrix)")

    # Initialize weights and bias
    # He initialization style scaling
    limit = cp.sqrt(6. / n_features)
    W = cp.random.uniform(-limit, limit, (n_features, n_targets), dtype=dtype)
    b = cp.zeros((1, n_targets), dtype=dtype)

    # Determine batch size
    if batch_size is None or batch_size > n_samples:
        batch_size = n_samples
        print("Using full batch gradient descent.")

    # Training loop
    print("Starting custom gradient descent training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        indices = cp.arange(n_samples)
        cp.random.shuffle(indices) # Shuffle data each epoch

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, n_samples)]
            X_batch = X_train_gpu[batch_indices]
            y_batch = y_train_gpu[batch_indices]
            current_batch_size = X_batch.shape[0]

            # --- Forward pass ---
            y_pred_batch = X_batch @ W + b

            # --- Calculate Cosine Similarity and Loss ---
            y_norm = cp.linalg.norm(y_batch, axis=1) + epsilon
            y_pred_norm = cp.linalg.norm(y_pred_batch, axis=1) + epsilon
            dot_product = cp.sum(y_pred_batch * y_batch, axis=1)

            cos_sim = dot_product / (y_pred_norm * y_norm)
            cos_sim = cp.clip(cos_sim, -1.0, 1.0) # Clip for stability

            # Loss = Mean of (1 - Cosine Similarity) over the batch
            batch_loss = cp.mean(1.0 - cos_sim)
            # Add L2 regularization to loss
            reg_loss = 0.5 * regularization * cp.sum(W*W)
            total_loss = batch_loss + reg_loss
            epoch_loss += total_loss * current_batch_size # Accumulate un-averaged loss

            # --- Backward pass (Calculate Gradients) ---
            # Gradient of loss L = 1 - cos_sim w.r.t y_pred_batch
            # dL/dy_pred = (cos_sim * y_pred - y) / (||y_pred|| * ||y||)

            # Reshape norms and cosine similarity for broadcasting: (batch_size, 1)
            y_norm_r = y_norm[:, cp.newaxis]
            y_pred_norm_r = y_pred_norm[:, cp.newaxis]
            cos_sim_r = cos_sim[:, cp.newaxis]

            dL_dy_pred = (cos_sim_r * y_pred_batch - y_batch) / (y_pred_norm_r * y_norm_r + epsilon)

            # Gradient w.r.t W and b (averaged over batch)
            grad_W = (X_batch.T @ dL_dy_pred) / current_batch_size
            grad_b = cp.mean(dL_dy_pred, axis=0) # Shape: (n_targets,)

            # Add L2 regularization gradient
            grad_W += regularization * W

            # --- Update weights ---
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b

        avg_epoch_loss = epoch_loss / n_samples
        if (epoch + 1) % 10 == 0 or epoch == 0: # Print loss every 10 epochs
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.6f}")

    print("Training finished.")

    # Predict on the Test Set
    print("Predicting on test set...")
    y_pred_test_gpu = X_test_gpu @ W + b

    # Evaluation
    print("Evaluating predictions...")
    # Note: y_test_gpu was potentially normalized above, so evaluation is consistent.
    r2_test, mse_test, cosine_similarities_test, low_sim_sample_info = evaluate_predictions(
        y_test_gpu, y_pred_test_gpu, test_sample_info, low_sim_threshold, use_gpu=True, matrix_type=matrix_type
    )

    # Return the trained weights and bias as the 'model' part of the package
    trained_model = {'W': W, 'b': b} # Keep model separate from package structure
    model_package = {
        'model': trained_model,
        'x_normalizer': x_normalizer,
        'y_normalizer': None, # Explicitly set y_normalizer to None
        'normalized_training': matrix_type == 'k_matrix' # Indicate X was normalized
    }
    return r2_test, mse_test, cosine_similarities_test, low_sim_sample_info, model_package

def train_and_evaluate_polynomial_cpu(X_train, X_test, y_train, y_test, test_sample_info=None, config=None):
    """
    Train a Polynomial Regression model with PCA -> Poly -> PCA on the CPU.
    Performs CPU normalization if matrix_type is k_matrix.

    Returns:
        r2_test, mse_test, cosine_similarities_test, low_sim_sample_info: Performance metrics and low similarity sample info
        model_package (dict): Dictionary containing the trained pipeline and normalizers ('model', 'x_normalizer', 'y_normalizer', 'normalized_training').
    """
    # Default config values
    pca1_n_components = config.get('pca1_n_components', 100) if config else 100
    pca2_n_components = config.get('pca2_n_components', 0.95) if config else 0.95
    poly_degree = config.get('poly_degree', 2) if config else 2
    low_sim_threshold = config.get('low_similarity_threshold', 0.85) if config else 0.85
    matrix_type = config.get('matrix_type', 'k_matrix') if config else 'k_matrix'
    x_normalizer = None # Initialize normalizers
    y_normalizer = None

    # Normalize data on CPU if k_matrix
    if matrix_type == 'k_matrix':
        print("Normalizing data (k_matrix) on CPU")
        x_normalizer = SklearnNormalizer()
        X_train = x_normalizer.fit_transform(X_train)
        X_test = x_normalizer.transform(X_test)

        y_normalizer = SklearnNormalizer()
        y_train = y_normalizer.fit_transform(y_train)
        y_test = y_normalizer.transform(y_test) # Normalize y_test for evaluation consistency
    else:
        print("Skipping normalization (not k_matrix)")

    # Create Pipeline: PCA -> PolynomialFeatures -> PCA -> LinearRegression
    print(f"Configuring pipeline: PCA({pca1_n_components}) -> Poly({poly_degree}) -> PCA({pca2_n_components}) -> Linear")
    pipeline = Pipeline([
        ('pca1', PCA(n_components=pca1_n_components)),
        ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ('pca2', PCA(n_components=pca2_n_components)),
        ('linear', LinearRegression())
    ])

    # Train the model pipeline
    print(f"Training pipeline on CPU...")
    pipeline.fit(X_train, y_train)
    print("Polynomial Regression with PCA model trained on CPU.")

    # Access PCA component details if needed (optional)
    pca1 = pipeline.named_steps['pca1']
    pca2 = pipeline.named_steps['pca2']
    print(f"PCA Step 1 retained {pca1.n_components_} components.")
    # The number of components for pca2 is only known after fitting/predicting
    # print(f"PCA Step 2 retained {pca2.n_components_} components.")

    # Predict on the Test Set
    y_pred_test = pipeline.predict(X_test)
    print("Predictions generated for the test set.")

    # Calculate metrics via evaluate_predictions
    y_test_np = np.array(y_test) # y_test might have been normalized
    y_pred_test_np = np.array(y_pred_test)

    r2_test, mse_test, cosine_similarities_test, low_sim_sample_info = evaluate_predictions(
        y_test_np, y_pred_test_np, test_sample_info, low_sim_threshold, use_gpu=False, matrix_type=matrix_type
    )

    model_package = {
        'model': pipeline, # The model is the entire pipeline
        'x_normalizer': x_normalizer,
        'y_normalizer': y_normalizer,
        'normalized_training': matrix_type == 'k_matrix'
    }

    return r2_test, mse_test, cosine_similarities_test, low_sim_sample_info, model_package

# Helper function for applying cuML PCA, handling float n_components
def apply_cuml_pca(X_gpu, n_components_target, step_name="PCA"):
    """Applies cuML PCA, handling float n_components for variance target."""
    print(f"Applying {step_name}...")
    if isinstance(n_components_target, float) and 0 < n_components_target < 1:
        # Initial fit with guessed components
        n_features = X_gpu.shape[1]
        if n_features <= 1: # Cannot apply PCA if only 1 feature
             print(f"  Skipping {step_name}: Only {n_features} feature(s).")
             return X_gpu, None # Return original data and no PCA object
        # Ensure n_components_int is at least 1
        n_components_int = min(n_features, max(1, int(n_features * n_components_target) if n_features > 1 else 1))
        print(f"  {step_name} initial components: {n_components_int} (target variance: {n_components_target*100:.1f}%)")
        pca = cuPCA(n_components=n_components_int)

        # Handle potential SVD convergence issues
        try:
            X_pca = pca.fit_transform(X_gpu)
        except Exception as e:
             print(f"  Error during {step_name} initial fit: {e}. Skipping PCA.")
             return X_gpu, None

        # Adjust components based on variance
        explained_var_ratio = pca.explained_variance_ratio_

        # Ensure explained_var_ratio is not None and not empty
        if explained_var_ratio is None or explained_var_ratio.size == 0:
             print(f"  {step_name}: Explained variance ratio not available after initial fit. Using initial components.")
             n_components_needed = pca.n_components_
        else:
            cumulative_var_ratio = cp.cumsum(explained_var_ratio)
            # Find the number of components needed
            if cp.any(cumulative_var_ratio >= n_components_target):
                n_components_needed = cp.argmax(cumulative_var_ratio >= n_components_target).item() + 1
            else:
                n_components_needed = n_features # Use all if target never reached

        n_components_needed = min(n_components_needed, n_features) # Clamp to max features

        # Refit if necessary and possible
        if n_components_needed != pca.n_components_ and n_components_needed < n_features:
            print(f"  Adjusting {step_name} to {n_components_needed} components to retain {n_components_target*100:.1f}% variance")
            pca = cuPCA(n_components=n_components_needed)
            try:
                 X_pca = pca.fit_transform(X_gpu)
            except Exception as e:
                 print(f"  Error during {step_name} refit: {e}. Using previous result.")
                 # Keep the result from the initial fit if refit fails
                 # Need to re-fit with the original n_components_int to get X_pca
                 pca = cuPCA(n_components=n_components_int)
                 try:
                     X_pca = pca.fit_transform(X_gpu)
                 except Exception as e_refit_orig:
                      print(f"  Error during {step_name} re-fit with initial components: {e_refit_orig}. Skipping PCA.")
                      return X_gpu, None # Fallback: skip PCA
        elif n_components_needed >= n_features:
             print(f"  Target variance requires >= available features ({n_features}). Using all {n_features} features for {step_name}.")
             X_pca = X_gpu
             pca = None # Indicate PCA wasn't effectively applied for reduction
        # else: # No need to print this if initial fit was sufficient
        #      print(f"  Initial {step_name} fit with {pca.n_components_} components already met variance target.")

    elif isinstance(n_components_target, int) and n_components_target >= X_gpu.shape[1]:
         print(f"  Skipping {step_name}: n_components ({n_components_target}) >= features ({X_gpu.shape[1]}).")
         return X_gpu, None
    elif X_gpu.shape[1] == 0:
         print(f"  Skipping {step_name}: Input has 0 features.")
         return X_gpu, None
    else: # Handle integer n_components or n_components=None
        # Clamp n_components_target if it's an int and larger than features
        if isinstance(n_components_target, int):
            n_components_actual = min(n_components_target, X_gpu.shape[1])
            if n_components_actual != n_components_target:
                 print(f"  Warning: Requested {n_components_target} components, but only {X_gpu.shape[1]} features available. Using {n_components_actual}.")
            n_components_target = n_components_actual
        elif n_components_target is None:
             n_components_target = X_gpu.shape[1] # Use all features if None

        pca = cuPCA(n_components=n_components_target)
        try:
            X_pca = pca.fit_transform(X_gpu)
        except Exception as e:
             print(f"  Error during {step_name} fit with {n_components_target} components: {e}. Skipping PCA.")
             return X_gpu, None

    if pca:
        print(f"  {step_name} retained {pca.n_components_} components.")
        if hasattr(pca, 'explained_variance_ratio_') and pca.explained_variance_ratio_ is not None:
             print(f"  Explained variance ratio sum: {cp.sum(pca.explained_variance_ratio_):.4f}")
        # else: # This case handled by the try-except and initial check now
        #      print(f"  Explained variance ratio not available for {step_name}.")
        return X_pca, pca
    else:
        return X_pca, None # Handle cases where PCA was skipped

def train_and_evaluate_polynomial_gpu(X_train, X_test, y_train, y_test, test_sample_info=None, config=None):
    """
    Train a Polynomial Regression model with PCA -> Poly -> PCA on the GPU.
    Refactored to use apply_cuml_pca and evaluate_predictions.
    Performs GPU normalization if matrix_type is k_matrix.

    Returns:
        r2_test, mse_test, cosine_similarities_test, low_sim_sample_info: Performance metrics and low similarity sample info
        model_package (dict): Dictionary containing the trained components ('pca1', 'poly_features', 'pca2', 'linear_model') and normalizers ('x_normalizer', 'y_normalizer', 'normalized_training').
    """
    # Default config values
    pca1_n_components = config.get('pca1_n_components', 100) if config else 100
    pca2_n_components = config.get('pca2_n_components', 0.95) if config else 0.95
    poly_degree = config.get('poly_degree', 2) if config else 2
    low_sim_threshold = config.get('low_similarity_threshold', 0.85) if config else 0.85
    matrix_type = config.get('matrix_type', 'k_matrix') if config else 'k_matrix'
    poly_interaction_only = config.get('poly_interaction_only', False) if config else False
    precision = config.get('precision', 'float64') # Default to float64
    x_normalizer = None # Initialize normalizers
    y_normalizer = None
    poly_features = None # Initialize poly_features
    pca1 = None
    pca2 = None
    model = None

    dtype = cp.float64 if precision == 'float64' else cp.float32
    print(f"Using {precision} precision for GPU computation.")

    # Convert data to CuPy arrays
    X_train_gpu = cp.asarray(X_train, dtype=dtype)
    y_train_gpu = cp.asarray(y_train, dtype=dtype)
    X_test_gpu = cp.asarray(X_test, dtype=dtype)
    y_test_gpu = cp.asarray(y_test, dtype=dtype) # Keep y_test on GPU for similarity calc

    # Normalize data on GPU if k_matrix
    if matrix_type == 'k_matrix':
        print("Normalizing data (k_matrix) on GPU")
        x_normalizer = cuMLNormalizer()
        X_train_gpu = x_normalizer.fit_transform(X_train_gpu)
        X_test_gpu = x_normalizer.transform(X_test_gpu)

        y_normalizer = cuMLNormalizer()
        y_train_gpu = y_normalizer.fit_transform(y_train_gpu)
        y_test_gpu = y_normalizer.transform(y_test_gpu) # Normalize y_test for evaluation consistency
    else:
        print("Skipping normalization (not k_matrix)")

    print(f"Training PolyReg (degree={poly_degree}) with PCA({pca1_n_components})->Poly->PCA({pca2_n_components}) on GPU...")
    print(f"Initial data shapes (after potential normalization): X_train={X_train_gpu.shape}, X_test={X_test_gpu.shape}")

    # --- Pipeline Steps ---
    # 1. First PCA
    X_train_processed, pca1 = apply_cuml_pca(X_train_gpu, pca1_n_components, step_name="PCA Step 1")
    if pca1:
        X_test_processed = pca1.transform(X_test_gpu)
    else:
        X_test_processed = X_test_gpu # Use original if PCA1 skipped
    print(f"After PCA Step 1: X_train={X_train_processed.shape}, X_test={X_test_processed.shape}")

    # 2. Polynomial Features
    if X_train_processed.shape[1] > 0:
        print("Generating Polynomial Features...")
        poly_features = cuPolynomialFeatures(degree=poly_degree, interaction_only=poly_interaction_only, include_bias=False, order='F') # Use Fortran order
        # Handle potential memory errors during fit_transform
        try:
             X_train_processed = poly_features.fit_transform(X_train_processed)
             X_test_processed = poly_features.transform(X_test_processed)
             print(f"After Polynomial Features: X_train={X_train_processed.shape}, X_test={X_test_processed.shape}")
        except cp.cuda.memory.OutOfMemoryError:
             print(f"Out of memory during Polynomial Features (degree={poly_degree}). Try reducing degree or using interaction_only=True.")
             # Cannot proceed, return default bad values and None for package
             return -1, float('inf'), cp.array([]), None, None
        except Exception as e:
             print(f"Error during Polynomial Features: {e}. Skipping.")
             poly_features = None # Set to None if skipped
             # Continue without polynomial features if another error occurs
             pass # X_train_processed and X_test_processed remain as they were after PCA1
    else:
        print("Skipping Polynomial Features: No features remaining after PCA Step 1.")
        poly_features = None # Set to None if skipped

    # 3. Second PCA
    X_train_processed, pca2 = apply_cuml_pca(X_train_processed, pca2_n_components, step_name="PCA Step 2")
    if pca2:
        X_test_processed = pca2.transform(X_test_processed)
    # else: X_test_processed remains from the previous step if PCA2 skipped
    print(f"After PCA Step 2: X_train={X_train_processed.shape}, X_test={X_test_processed.shape}")

    # 4. Linear Regression
    print("Training Linear Regression model on GPU...")
    if X_train_processed.shape[1] == 0:
         print("Skipping Linear Regression: No features remaining after preprocessing.")
         # Handle case with no features: predict zeros or mean? Predicting zeros.
         y_pred_test_gpu = cp.zeros_like(y_test_gpu)
         model = None # No model trained
    else:
        # Ensure y_train_gpu is C-contiguous if it's multi-output
        if y_train_gpu.ndim > 1 and y_train_gpu.shape[1] > 1 and not y_train_gpu.flags['C_CONTIGUOUS']:
            y_train_gpu = cp.ascontiguousarray(y_train_gpu)

        model = cuLinearRegression(fit_intercept=True, algorithm='svd') # SVD is often more stable
        try:
             model.fit(X_train_processed, y_train_gpu)
             print("Linear Regression model trained.")
             # Predict on the Test Set
             y_pred_test_gpu = model.predict(X_test_processed)
             print("Predictions generated for the test set.")
        except Exception as e:
             print(f"Error during Linear Regression fit/predict: {e}")
             # Return default bad values if model fails, package is None
             return -1, float('inf'), cp.array([]), None, None

    # Evaluation
    print("Evaluating predictions...")
    r2_test, mse_test, cosine_similarities_test, low_sim_sample_info = evaluate_predictions(
        y_test_gpu, y_pred_test_gpu, test_sample_info, low_sim_threshold, use_gpu=True, matrix_type=matrix_type
    )

    # Create a dictionary to hold the trained components (PCA, Poly, Model)
    # This acts as the 'model' part of the package
    trained_components = {
        'pca1': pca1,
        'poly_features': poly_features,
        'pca2': pca2,
        'linear_model': model
    }
    model_package = {
        'model': trained_components,
        'x_normalizer': x_normalizer,
        'y_normalizer': None, # Explicitly set y_normalizer to None
        'normalized_training': matrix_type == 'k_matrix' # Indicate X was normalized
    }

    return r2_test, mse_test, cosine_similarities_test, low_sim_sample_info, model_package

# Shared evaluation logic (can be used by CPU and GPU models)
def evaluate_predictions(y_true, y_pred, test_sample_info, low_sim_threshold, use_gpu=False, matrix_type='k_matrix'):
    """
    Calculates R2, MSE, and Cosine Similarity, identifies low similarity samples.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        test_sample_info: DataFrame with sample information
        low_sim_threshold: Threshold for identifying low similarity samples
        use_gpu: Whether to use GPU for computations
        matrix_type: Type of matrix ('k_matrix' or 'v_matrix') - MSE is primary for v_matrix
        
    Returns:
        r2, mse, cosine_similarities, low_sim_sample_info_df: Performance metrics and low similarity samples
    """
    if use_gpu:
        y_true_np = cp.asnumpy(y_true)
        y_pred_np = cp.asnumpy(y_pred)
        r2_func = cu_r2_score
        mse_func = cu_mean_squared_error
        lib = cp # Use cupy for calculations
    else:
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        r2_func = r2_score
        mse_func = mean_squared_error
        lib = np # Use numpy for calculations

    print("r2_func: ", r2_func)
    # Calculate metrics
    # Need numpy arrays for sklearn/cuml metrics
    r2 = r2_func(y_true_np, y_pred_np)
    mse = mse_func(y_true_np, y_pred_np)
    
    # Always calculate Cosine Similarity for both k_matrix and v_matrix
    # --- Debugging Checks --- 
    if lib.any(lib.isnan(y_true)) or lib.any(lib.isinf(y_true)):
        print("Warning: NaNs or Infs found in y_true before cosine similarity calculation.")
    if lib.any(lib.isnan(y_pred)) or lib.any(lib.isinf(y_pred)):
        print("Warning: NaNs or Infs found in y_pred before cosine similarity calculation.")
        # Optional: Print indices or count
        nan_inf_pred_count = lib.sum(lib.isnan(y_pred) | lib.isinf(y_pred))
        print(f"  Number of NaN/Inf values in y_pred: {nan_inf_pred_count}")
        # Consider saving or inspecting these problematic y_pred vectors
    # --- End Debugging Checks ---
    
    # Calculate Cosine Similarity
    dot_products = lib.sum(y_true * y_pred, axis=1)
    norm_target = lib.linalg.norm(y_true, axis=1)
    norm_pred = lib.linalg.norm(y_pred, axis=1)
    epsilon = 1e-9
    
    # --- Debugging Checks for Zero Norms ---
    zero_norm_target_indices = lib.where(norm_target < epsilon)[0]
    if zero_norm_target_indices.size > 0:
        print(f"Warning: {zero_norm_target_indices.size} samples found with near-zero norm in y_true.")
        # print(f"  Indices: {zero_norm_target_indices[:10]}") # Print first few indices
    
    zero_norm_pred_indices = lib.where(norm_pred < epsilon)[0]
    if zero_norm_pred_indices.size > 0:
        print(f"Warning: {zero_norm_pred_indices.size} samples found with near-zero norm in y_pred.")
        # print(f"  Indices: {zero_norm_pred_indices[:10]}") # Print first few indices
    # --- End Debugging Checks ---
    
    denominator = norm_target * norm_pred + epsilon
    # Use np.errstate to suppress expected divide-by-zero warnings if we handle NaNs later
    # with np.errstate(divide='ignore', invalid='ignore'): # REMOVED errstate context for divide
    #     cosine_similarities = lib.divide(dot_products, denominator, where=denominator != 0) 
    #     # Ensure result is still a cupy array if lib is cupy
    #     if lib == cp and not isinstance(cosine_similarities, cp.ndarray):
    #         # This can happen if lib.divide falls back to numpy behaviour in edge cases
    #         cosine_similarities = cp.asarray(cosine_similarities)
            
    # Calculate safe division mask
    safe_division_mask = denominator > epsilon # Use epsilon directly
    
    # Initialize similarities array (using appropriate library)
    cosine_similarities = lib.zeros_like(dot_products, dtype=dot_products.dtype)
    
    # Perform division only where safe, suppressing warnings temporarily for this operation
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate division for safe elements only
        safe_dot_products = dot_products[safe_division_mask]
        safe_denominators = denominator[safe_division_mask]
        cosine_similarities[safe_division_mask] = lib.divide(safe_dot_products, safe_denominators)

    # cosine_similarities where denominator was unsafe remain 0 (or initial value)
            
    # Clip results AFTER division
    cosine_similarities = lib.clip(cosine_similarities, -1.0, 1.0)
    
    # Final check for NaNs after calculation
    nan_count_after = lib.sum(lib.isnan(cosine_similarities))
    if nan_count_after > 0:
        print(f"Warning: {nan_count_after} NaNs generated during cosine similarity calculation (potentially from sources other than zero denominator).")

    # Convert similarities to numpy for indexing if needed
    cosine_similarities_np = cp.asnumpy(cosine_similarities) if use_gpu else cosine_similarities
    
    if matrix_type == 'v_matrix':
        print("Note: For v_matrix, MSE is the primary metric but cosine similarity is still reported")

    # Identify low similarity samples
    low_sim_sample_info_df = None
    
    if test_sample_info is not None and len(cosine_similarities_np) > 0:
        # Exclude NaNs when identifying low similarity samples
        valid_indices = ~np.isnan(cosine_similarities_np)
        low_sim_mask = (cosine_similarities_np[valid_indices] < low_sim_threshold)
        original_indices = np.arange(len(cosine_similarities_np))
        low_sim_indices = original_indices[valid_indices][low_sim_mask]
        
        if len(low_sim_indices) > 0:
            # Check if test_sample_info is a Pandas DataFrame before using .iloc
            if test_sample_info is not None and isinstance(test_sample_info, pd.DataFrame):
                # Ensure indices are within bounds (can happen with empty test set edge cases)
                valid_low_sim_indices = low_sim_indices[low_sim_indices < len(test_sample_info)]
                if len(valid_low_sim_indices) > 0:
                     low_sim_sample_info_df = test_sample_info.iloc[valid_low_sim_indices].copy()
                     # Assign similarity values only for the identified low-similarity indices
                     low_sim_sample_info_df['cosine_similarity'] = cosine_similarities_np[valid_low_sim_indices]
                else:
                    print("Warning: Low similarity indices out of bounds for test_sample_info.")
                    low_sim_sample_info_df = None # Reset if indices were invalid

            else:
                print("Warning: test_sample_info is not a Pandas DataFrame or is None, cannot extract low similarity sample details.")
                low_sim_sample_info_df = None # Ensure it's None if info isn't DataFrame

    # Return the original similarities array (cupy or numpy), potentially containing NaNs
    return r2, mse, cosine_similarities, low_sim_sample_info_df

def train_and_evaluate_lasso(X_train, X_test, y_train, y_test, test_sample_info=None, config=None, use_gpu=False):
    """
    Train a Lasso Regression model and evaluate its performance.
    Can run on CPU (sklearn) or GPU (cuML).
    Performs normalization based on use_gpu and matrix_type.

    Returns:
        r2_test, mse_test, cosine_similarities_test, low_sim_sample_info: Performance metrics and low similarity sample info
        model_package (dict): Dictionary containing the trained model and normalizers ('model', 'x_normalizer', 'y_normalizer', 'normalized_training').
    """
    alpha = config.get('alpha', 1.0) if config else 1.0
    low_sim_threshold = config.get('low_similarity_threshold', 0.85) if config else 0.85
    matrix_type = config.get('matrix_type', 'k_matrix') if config else 'k_matrix'
    precision = config.get('precision', 'float64') if use_gpu else 'float64' # GPU precision from config
    x_normalizer = None # Initialize normalizers
    y_normalizer = None

    if use_gpu:
        print(f"Training Lasso (alpha={alpha}) on GPU using {precision} precision...")
        dtype = cp.float64 if precision == 'float64' else cp.float32
        X_train_dev = cp.asarray(X_train, dtype=dtype)
        y_train_dev = cp.asarray(y_train, dtype=dtype)
        X_test_dev = cp.asarray(X_test, dtype=dtype)
        y_test_dev = cp.asarray(y_test, dtype=dtype) # Keep y_test on device for evaluation

        # Normalize data on GPU if k_matrix
        if matrix_type == 'k_matrix':
            print("Normalizing data (k_matrix) on GPU")
            x_normalizer = cuMLNormalizer()
            X_train_dev = x_normalizer.fit_transform(X_train_dev)
            X_test_dev = x_normalizer.transform(X_test_dev)

            y_normalizer = cuMLNormalizer()
            y_train_dev = y_normalizer.fit_transform(y_train_dev)
            y_test_dev = y_normalizer.transform(y_test_dev) # Normalize y_test for evaluation consistency
        else:
            print("Skipping normalization (not k_matrix)")

        # Ensure y_train_dev is C-contiguous for cuML
        if y_train_dev.ndim > 1 and y_train_dev.shape[1] > 1 and not y_train_dev.flags['C_CONTIGUOUS']:
            y_train_dev = cp.ascontiguousarray(y_train_dev)

        model = cuLasso(alpha=alpha, fit_intercept=True)
        eval_lib = cp
    else:
        print(f"Training Lasso (alpha={alpha}) on CPU...")
        # Use numpy arrays for CPU
        X_train_dev, y_train_dev = np.array(X_train), np.array(y_train)
        X_test_dev, y_test_dev = np.array(X_test), np.array(y_test)

        # Normalize data on CPU if k_matrix
        if matrix_type == 'k_matrix':
            print("Normalizing data (k_matrix) on CPU")
            x_normalizer = SklearnNormalizer()
            X_train_dev = x_normalizer.fit_transform(X_train_dev)
            X_test_dev = x_normalizer.transform(X_test_dev)

            y_normalizer = SklearnNormalizer()
            y_train_dev = y_normalizer.fit_transform(y_train_dev)
            y_test_dev = y_normalizer.transform(y_test_dev) # Normalize y_test for evaluation consistency
        else:
            print("Skipping normalization (not k_matrix)")

        model = Lasso(alpha=alpha, fit_intercept=True)
        eval_lib = np

    # Train Model
    try:
        model.fit(X_train_dev, y_train_dev)
        print(f"Lasso model trained {'on GPU' if use_gpu else 'on CPU'}.")
        # Predict on the Test Set
        y_pred_test_dev = model.predict(X_test_dev)
        print("Predictions generated for the test set.")
    except Exception as e:
        print(f"Error during Lasso {'GPU' if use_gpu else 'CPU'} fit/predict: {e}")
        # Return appropriately typed empty array and None package
        empty_arr = cp.array([]) if use_gpu else np.array([])
        return -1, float('inf'), empty_arr, None, None # Return default bad values

    # Evaluation
    print("Evaluating predictions...")
    r2_test, mse_test, cosine_similarities_test, low_sim_sample_info = evaluate_predictions(
        y_test_dev, y_pred_test_dev, test_sample_info, low_sim_threshold, use_gpu=use_gpu, matrix_type=matrix_type
    )

    model_package = {
        'model': model,
        'x_normalizer': x_normalizer,
        'y_normalizer': y_normalizer,
        'normalized_training': matrix_type == 'k_matrix'
    }

    return r2_test, mse_test, cosine_similarities_test, low_sim_sample_info, model_package

def train_and_evaluate_ridge(X_train, X_test, y_train, y_test, test_sample_info=None, config=None, use_gpu=False):
    """
    Train a Ridge Regression model and evaluate its performance.
    Can run on CPU (sklearn) or GPU (cuML).
    Performs normalization based on use_gpu and matrix_type.

    Returns:
        r2_test, mse_test, cosine_similarities_test, low_sim_sample_info: Performance metrics and low similarity sample info
        model_package (dict): Dictionary containing the trained model and normalizers ('model', 'x_normalizer', 'y_normalizer', 'normalized_training').
    """
    alpha = config.get('alpha', 1.0) if config else 1.0
    low_sim_threshold = config.get('low_similarity_threshold', 0.85) if config else 0.85
    matrix_type = config.get('matrix_type', 'k_matrix') if config else 'k_matrix'
    precision = config.get('precision', 'float64') if use_gpu else 'float64' # GPU precision from config
    x_normalizer = None # Initialize normalizers
    y_normalizer = None

    if use_gpu:
        print(f"Training Ridge (alpha={alpha}) on GPU using {precision} precision...")
        dtype = cp.float64 if precision == 'float64' else cp.float32
        X_train_dev = cp.asarray(X_train, dtype=dtype)
        y_train_dev = cp.asarray(y_train, dtype=dtype)
        X_test_dev = cp.asarray(X_test, dtype=dtype)
        y_test_dev = cp.asarray(y_test, dtype=dtype) # Keep y_test on device for evaluation

        # Normalize data on GPU if k_matrix
        if matrix_type == 'k_matrix':
            print("Normalizing data (k_matrix) on GPU")
            x_normalizer = cuMLNormalizer()
            X_train_dev = x_normalizer.fit_transform(X_train_dev)
            X_test_dev = x_normalizer.transform(X_test_dev)

            y_normalizer = cuMLNormalizer()
            y_train_dev = y_normalizer.fit_transform(y_train_dev)
            y_test_dev = y_normalizer.transform(y_test_dev) # Normalize y_test for evaluation consistency
        else:
            print("Skipping normalization (not k_matrix)")

        # Ensure y_train_dev is C-contiguous for cuML
        if y_train_dev.ndim > 1 and y_train_dev.shape[1] > 1 and not y_train_dev.flags['C_CONTIGUOUS']:
            y_train_dev = cp.ascontiguousarray(y_train_dev)

        # cuML Ridge has different solver options
        model = cuRidge(alpha=alpha, fit_intercept=True, solver='eig') # 'eig' is common, 'svd' also available
        eval_lib = cp
    else:
        print(f"Training Ridge (alpha={alpha}) on CPU...")
        # Use numpy arrays for CPU
        X_train_dev, y_train_dev = np.array(X_train), np.array(y_train)
        X_test_dev, y_test_dev = np.array(X_test), np.array(y_test)

        # Normalize data on CPU if k_matrix
        if matrix_type == 'k_matrix':
            print("Normalizing data (k_matrix) on CPU")
            x_normalizer = SklearnNormalizer()
            X_train_dev = x_normalizer.fit_transform(X_train_dev)
            X_test_dev = x_normalizer.transform(X_test_dev)

            y_normalizer = SklearnNormalizer()
            y_train_dev = y_normalizer.fit_transform(y_train_dev)
            y_test_dev = y_normalizer.transform(y_test_dev) # Normalize y_test for evaluation consistency
        else:
            print("Skipping normalization (not k_matrix)")

        model = Ridge(alpha=alpha, fit_intercept=True)
        eval_lib = np

    # Train Model
    try:
        model.fit(X_train_dev, y_train_dev)
        print(f"Ridge model trained {'on GPU' if use_gpu else 'on CPU'}.")
        # Predict on the Test Set
        y_pred_test_dev = model.predict(X_test_dev)
        print("Predictions generated for the test set.")
    except Exception as e:
        print(f"Error during Ridge {'GPU' if use_gpu else 'CPU'} fit/predict: {e}")
        # Return appropriately typed empty array and None package
        empty_arr = cp.array([]) if use_gpu else np.array([])
        return -1, float('inf'), empty_arr, None, None # Return default bad values

    # Evaluation
    print("Evaluating predictions...")
    r2_test, mse_test, cosine_similarities_test, low_sim_sample_info = evaluate_predictions(
        y_test_dev, y_pred_test_dev, test_sample_info, low_sim_threshold, use_gpu=use_gpu, matrix_type=matrix_type
    )

    model_package = {
        'model': model,
        'x_normalizer': x_normalizer,
        'y_normalizer': y_normalizer,
        'normalized_training': matrix_type == 'k_matrix'
    }

    return r2_test, mse_test, cosine_similarities_test, low_sim_sample_info, model_package

# ==================================
# PyTorch Linear Regression Model (GPU)
# ==================================

class TorchLinearRegressionGPU(nn.Module):
    """
    Simple Linear Regression model using PyTorch for GPU acceleration.
    """
    def __init__(self, n_features, n_targets, dtype=torch.float64):
        super().__init__()
        print(f"TORCH DTYPE: {dtype}")
        self.linear = nn.Linear(n_features, n_targets, bias=True, dtype=dtype)

    def forward(self, x):
        return self.linear(x)

def train_and_evaluate_torch_linear_gpu(X_train, X_test, y_train, y_test, test_sample_info=None, config=None):
    """
    Train a Linear Regression model on the GPU using PyTorch and evaluate its performance.
    Performs single-shot closed-form regression without regularization.
    """
    # Configuration parameters
    matrix_type = config.get('matrix_type', 'k_matrix') if config else 'k_matrix'
    low_sim_threshold = config.get('low_similarity_threshold', 0.85) if config else 0.85
    precision = config.get('precision', 'float64')

    # Determine PyTorch dtype
    if precision == 'float64':
        dtype = torch.float64
    elif precision == 'float16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input data to tensors
    print(f"Device: {device}")
    X_train_torch = torch.from_numpy(X_train).to(device=device, dtype=dtype)
    y_train_torch = torch.from_numpy(y_train).to(device=device, dtype=dtype)
    X_test_torch = torch.from_numpy(X_test).to(device=device, dtype=dtype)
    y_test_torch = torch.from_numpy(y_test).to(device=device, dtype=dtype)

    print(f"X_train_torch: {X_train_torch.shape}")
    print(f"y_train_torch: {y_train_torch.shape}")
    print(f"X_test_torch: {X_test_torch.shape}")
    print(f"y_test_torch: {y_test_torch.shape}")

    # Ensure y tensors have shape (N, T)
    if y_train_torch.ndim == 1:
        y_train_torch = y_train_torch.unsqueeze(1)
        y_test_torch = y_test_torch.unsqueeze(1)

    # Build design matrix with bias term
    ones = torch.ones(X_train_torch.size(0), 1, device=device, dtype=dtype)
    X_aug = torch.cat([X_train_torch, ones], dim=1)

    # Closed-form solution via pseudoinverse
    pinv = torch.pinverse(X_aug)
    coeff = pinv @ y_train_torch         # shape (F+1, T)
    W = coeff[:-1, :]                     # weights
    b = coeff[-1, :]                      # bias

    # Predict on the test set
    y_pred_test_torch = X_test_torch @ W + b

    # Convert tensors to CuPy arrays for evaluation
    y_test_gpu = cp.asarray(y_test_torch.cpu())
    y_pred_test_gpu = cp.asarray(y_pred_test_torch.cpu())

    # Evaluate predictions
    r2_test, mse_test, cosine_similarities_test, low_sim_sample_info = evaluate_predictions(
        y_test_gpu, y_pred_test_gpu, test_sample_info, low_sim_threshold, use_gpu=True, matrix_type=matrix_type
    )

    # Package the learned parameters as PyTorch tensors
    model_package = {
        'weights': W.detach().cpu(),  # torch.Tensor
        'bias': b.detach().cpu(),     # torch.Tensor
        'normalized_training': False
    }

    return r2_test, mse_test, cosine_similarities_test, low_sim_sample_info, model_package

# --- Add other model implementations below ---
# e.g., def train_and_evaluate_ridge(...):
# e.g., def train_and_evaluate_nn(...): 