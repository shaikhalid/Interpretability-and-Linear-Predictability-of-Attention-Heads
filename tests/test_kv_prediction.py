import subprocess
import re
import os
import unittest

# Define paths relative to the script's location or workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Model configurations
MODELS = {
    'llama3': {
        'pipeline_script': os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/pipeline_test_llama3.sh'),
        'baseline_log': os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/logs/Baseline_winogrande_l8b.log'),
        'prediction_log': os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/logs/Prediction_winogrande_l8b.log'),
        'alias': 'l8b'
    },
    'qwen3': {
        'pipeline_script': os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/pipeline_test_qwen3.sh'),
        'baseline_log': os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/logs/Baseline_winogrande_q32b.log'),
        'prediction_log': os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/logs/Prediction_winogrande_q32b.log'),
        'alias': 'q32b'
    }
}

def run_pipeline_script(model_name):
    """Runs the KV prediction pipeline script for a specific model."""
    model_config = MODELS[model_name]
    pipeline_script = model_config['pipeline_script']
    
    print(f"Running {model_name} pipeline script: {pipeline_script}...")
    try:
        # Ensure the script is executable
        subprocess.run(['chmod', '+x', pipeline_script], check=True, cwd=WORKSPACE_ROOT)
        # Run the script using bash explicitly
        result = subprocess.run(['bash', pipeline_script], capture_output=True, text=True, check=True, cwd=WORKSPACE_ROOT)
        print(f"{model_name} pipeline script executed successfully.")
        print("Stdout:")
        print(result.stdout)
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {model_name} pipeline script: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: {model_name} pipeline script not found at {pipeline_script}")
        raise

def extract_accuracy_from_log(log_file_path):
    """Extracts accuracy from a given log file."""
    print(f"Extracting accuracy from {log_file_path}...")
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

        # Try patterns in order of specificity, from most specific to least specific
        
        # Special format for winogrande logs: "acc,none: 0.7387529597474349"
        match = re.search(r'acc,none:\s*([0-9]+\.[0-9]+)', content, re.IGNORECASE)
        if match:
            accuracy = float(match.group(1)) * 100  # Convert to percentage
            print(f"Found accuracy (acc,none format): {accuracy}% (converted from {match.group(1)})")
            return accuracy
        
        # Look for "accuracy: X.XX" or "acc: X.XX" patterns (with decimal)
        match = re.search(r'(?:accuracy|acc)(?:[:,]|\s+is:?\s+)?\s*([0-9]+\.[0-9]+)', content, re.IGNORECASE)
        if match:
            # Check if it's already a percentage (0-100) or a decimal (0-1)
            value = float(match.group(1))
            if value < 1.0:  # Likely a decimal between 0-1
                accuracy = value * 100
                print(f"Found accuracy (decimal format): {accuracy}% (converted from {value})")
            else:  # Likely already a percentage
                accuracy = value
                print(f"Found accuracy (percentage format): {accuracy}%")
            return accuracy
            
        # Look for "accuracy: XX" or "acc: XX" patterns (whole numbers, no decimal)
        match = re.search(r'(?:accuracy|acc)(?:[:,]|\s+is:?\s+)?\s*([0-9]+)(?!\.[0-9])', content, re.IGNORECASE)
        if match:
            # Avoid matching things like 'access...4' by checking if word boundary comes before
            matched_text = match.group(0)
            if re.search(r'\baccuracy|\bacc', matched_text, re.IGNORECASE):
                accuracy = float(match.group(1))
                print(f"Found accuracy (whole number): {accuracy}%")
                return accuracy

        print(f"Accuracy not found in the expected format in {log_file_path}.")
        print("File content preview (truncated):")
        print(content[:500])
        return None
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing log file {log_file_path}: {e}")
        return None

class TestKVPredictionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run the pipeline scripts for all models before tests."""
        print("Running pipelines from scratch - deleting existing log files if they exist.")
        
        # Clean up all log files for all models
        for model_name, model_config in MODELS.items():
            baseline_log = model_config['baseline_log']
            prediction_log = model_config['prediction_log']
            
            for log_file in [baseline_log, prediction_log]:
                if os.path.exists(log_file):
                    print(f"Removing {model_name} log file: {log_file}")
                    os.remove(log_file)
        
        # Run pipelines for all models
        for model_name in MODELS.keys():
            print(f"\n=== Running {model_name} Pipeline ===")
            run_pipeline_script(model_name)

    def test_llama3_baseline_accuracy(self):
        """Tests if the llama3 baseline accuracy is 70+."""
        print("\n--- Testing LLaMA3 Baseline Accuracy ---")
        model_config = MODELS['llama3']
        baseline_accuracy = extract_accuracy_from_log(model_config['baseline_log'])
        self.assertIsNotNone(baseline_accuracy, f"Could not extract llama3 baseline accuracy from {model_config['baseline_log']}")
        print(f"LLaMA3 baseline accuracy extracted: {baseline_accuracy}")
        self.assertGreaterEqual(baseline_accuracy, 70.0, f"LLaMA3 baseline accuracy {baseline_accuracy} is below 70")
        print(f"LLaMA3 baseline accuracy test passed ({baseline_accuracy} >= 70.0)")

    def test_llama3_prediction_accuracy_deviation(self):
        """Tests if the llama3 prediction accuracy is within +/-15 of baseline accuracy."""
        print("\n--- Testing LLaMA3 Prediction Accuracy Deviation ---")
        model_config = MODELS['llama3']
        baseline_accuracy = extract_accuracy_from_log(model_config['baseline_log'])
        self.assertIsNotNone(baseline_accuracy, f"Could not extract llama3 baseline accuracy from {model_config['baseline_log']} for deviation test")
        
        prediction_accuracy = extract_accuracy_from_log(model_config['prediction_log'])
        self.assertIsNotNone(prediction_accuracy, f"Could not extract llama3 prediction accuracy from {model_config['prediction_log']}")

        print(f"LLaMA3 baseline accuracy for deviation check: {baseline_accuracy}")
        print(f"LLaMA3 prediction accuracy for deviation check: {prediction_accuracy}")
        
        deviation = abs(prediction_accuracy - baseline_accuracy)
        print(f"LLaMA3 deviation: |{prediction_accuracy} - {baseline_accuracy}| = {deviation}")
        
        self.assertLessEqual(deviation, 15.0, f"LLaMA3 prediction accuracy deviation {deviation} is more than 15 from baseline {baseline_accuracy}")
        print(f"LLaMA3 prediction accuracy deviation test passed ({deviation} <= 15.0)")

    def test_qwen3_baseline_accuracy(self):
        """Tests if the qwen3 baseline accuracy is 70+."""
        print("\n--- Testing Qwen3 Baseline Accuracy ---")
        model_config = MODELS['qwen3']
        baseline_accuracy = extract_accuracy_from_log(model_config['baseline_log'])
        self.assertIsNotNone(baseline_accuracy, f"Could not extract qwen3 baseline accuracy from {model_config['baseline_log']}")
        print(f"Qwen3 baseline accuracy extracted: {baseline_accuracy}")
        self.assertGreaterEqual(baseline_accuracy, 70.0, f"Qwen3 baseline accuracy {baseline_accuracy} is below 70")
        print(f"Qwen3 baseline accuracy test passed ({baseline_accuracy} >= 70.0)")

    def test_qwen3_prediction_accuracy_deviation(self):
        """Tests if the qwen3 prediction accuracy is within +/-15 of baseline accuracy."""
        print("\n--- Testing Qwen3 Prediction Accuracy Deviation ---")
        model_config = MODELS['qwen3']
        baseline_accuracy = extract_accuracy_from_log(model_config['baseline_log'])
        self.assertIsNotNone(baseline_accuracy, f"Could not extract qwen3 baseline accuracy from {model_config['baseline_log']} for deviation test")
        
        prediction_accuracy = extract_accuracy_from_log(model_config['prediction_log'])
        self.assertIsNotNone(prediction_accuracy, f"Could not extract qwen3 prediction accuracy from {model_config['prediction_log']}")

        print(f"Qwen3 baseline accuracy for deviation check: {baseline_accuracy}")
        print(f"Qwen3 prediction accuracy for deviation check: {prediction_accuracy}")
        
        deviation = abs(prediction_accuracy - baseline_accuracy)
        print(f"Qwen3 deviation: |{prediction_accuracy} - {baseline_accuracy}| = {deviation}")
        
        self.assertLessEqual(deviation, 15.0, f"Qwen3 prediction accuracy deviation {deviation} is more than 15 from baseline {baseline_accuracy}")
        print(f"Qwen3 prediction accuracy deviation test passed ({deviation} <= 15.0)")

if __name__ == '__main__':
    print("Starting KV Prediction Pipeline Test Suite for LLaMA3 and Qwen3...")
    
    # Check if log directories exist for all models, create if not
    for model_name, model_config in MODELS.items():
        log_dir = os.path.dirname(model_config['baseline_log'])
        if not os.path.exists(log_dir):
            print(f"Log directory {log_dir} not found. Attempting to create.")
            try:
                os.makedirs(log_dir, exist_ok=True)
                print(f"Log directory {log_dir} created.")
            except Exception as e:
                print(f"Failed to create log directory {log_dir}: {e}")

    unittest.main() 