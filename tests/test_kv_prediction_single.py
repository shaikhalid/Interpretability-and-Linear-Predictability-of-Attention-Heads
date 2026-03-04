import unittest
import re
import os

# Define paths relative to the script's location or workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASELINE_LOG_PATH = os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/logs/Baseline_winogrande_l8b.log')
PREDICTION_LOG_PATH = os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/logs/Prediction_winogrande_l8b.log')

def extract_accuracy_from_log(log_file_path):
    """Extracts accuracy from a given log file."""
    print(f"Extracting accuracy from {log_file_path}...")
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

        # Try patterns in order of specificity, from most specific to least specific
        
        # Special format for winogrande logs: "acc,none: 0.7387529597474349"
        print('Trying "acc,none:" pattern first (most specific)')
        match = re.search(r'acc,none:\s*([0-9]+\.[0-9]+)', content, re.IGNORECASE)
        if match:
            print(f"Found acc,none pattern: '{match.group(0)}' with value {match.group(1)}")
            accuracy = float(match.group(1)) * 100  # Convert to percentage
            print(f"Extracted accuracy: {accuracy}% (converted from {match.group(1)})")
            return accuracy
        
        # Try other patterns if the specific one didn't match
        print('Now trying more generic patterns...')
        
        # Look for "accuracy: X.XX" or "acc: X.XX" patterns (with decimal)
        match = re.search(r'(?:accuracy|acc)(?:[:,]|\s+is:?\s+)?\s*([0-9]+\.[0-9]+)', content, re.IGNORECASE)
        if match:
            print(f"Found standard accuracy pattern: '{match.group(0)}' with value {match.group(1)}")
            # Check if it's already a percentage (0-100) or a decimal (0-1)
            value = float(match.group(1))
            if value < 1.0:  # Likely a decimal between 0-1
                accuracy = value * 100
                print(f"Extracted accuracy: {accuracy}% (converted from decimal {value})")
            else:  # Likely already a percentage
                accuracy = value
                print(f"Extracted accuracy: {accuracy}%")
            return accuracy
            
        # Look for "accuracy: XX" or "acc: XX" patterns (whole numbers, no decimal)
        match = re.search(r'(?:accuracy|acc)(?:[:,]|\s+is:?\s+)?\s*([0-9]+)(?!\.[0-9])', content, re.IGNORECASE)
        if match:
            # Avoid matching things like 'access...4' by checking if word boundary comes before
            matched_text = match.group(0)
            if re.search(r'\baccuracy|\bacc', matched_text, re.IGNORECASE):
                print(f"Found whole number accuracy pattern: '{matched_text}' with value {match.group(1)}")
                accuracy = float(match.group(1))
                print(f"Extracted accuracy: {accuracy}%")
                return accuracy
            else:
                print(f"Skipping likely false match: '{matched_text}'")

        # No matches found
        print(f"Accuracy not found in the expected format in {log_file_path}.")
        # Print a small portion of the file for debugging
        print("File content preview (truncated):")
        print(content[:500]) 
        return None
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing log file {log_file_path}: {e}")
        return None

class TestBaselineAccuracy(unittest.TestCase):
    
    def test_baseline_accuracy(self):
        """Tests if the baseline accuracy is 70+."""
        print("\n--- Testing Baseline Accuracy ---")
        baseline_accuracy = extract_accuracy_from_log(BASELINE_LOG_PATH)
        self.assertIsNotNone(baseline_accuracy, f"Could not extract baseline accuracy from {BASELINE_LOG_PATH}")
        print(f"Baseline accuracy extracted: {baseline_accuracy}")
        self.assertGreaterEqual(baseline_accuracy, 70.0, f"Baseline accuracy {baseline_accuracy} is below 70")
        print(f"Baseline accuracy test passed ({baseline_accuracy} >= 70.0)")

if __name__ == '__main__':
    unittest.main() 