import re
import os

# Define paths relative to the script's location or workspace root
# This assumes the debug script is in the 'tests' directory
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# *** IMPORTANT: Make sure this path points to your ACTUAL baseline log file ***
# or create a dummy file at this location with the content you want to test.
BASELINE_LOG_PATH = os.path.join(WORKSPACE_ROOT, 'sbatch_files/kv_prediction/logs/Baseline_winogrande_l8b.log')

# --- Paste the extract_accuracy_from_log function here ---
# (Copied from tests/test_kv_prediction.py and adapted slightly for standalone use if needed)
def extract_accuracy_from_log(log_file_path):
    """Extracts accuracy from a given log file."""
    print(f"Attempting to extract accuracy from: {log_file_path}")
    if not os.path.exists(log_file_path):
        print(f"ERROR: Log file not found at {log_file_path}")
        print("Please ensure the file exists or create a dummy file with sample content for testing.")
        # Create a dummy log file with placeholder content if it doesn't exist,
        # so the script can run for demonstration. User should replace this.
        print(f"Creating a dummy log file at {log_file_path} for demonstration.")
        print("YOU SHOULD REPLACE THE CONTENT of this dummy file with your actual log data.")
        dummy_content = """
        Some initial log lines...
        Another log line.
        Final Test Acc: 73.5%
        Some other statistics...
        accuracy: 0.732
        model_accuracy = 75.2
        Overall acc is 78.9
        """
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, 'w') as f_dummy:
                f_dummy.write(dummy_content)
            print(f"Dummy file created at {log_file_path} with placeholder content.")
            print("Please edit it with your actual log format and rerun this script.")
        except Exception as e_dummy:
            print(f"Could not create dummy file: {e_dummy}")
            return None # Exit if we can't even make a dummy file

    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        print(f"--- Content of {log_file_path} (first 500 chars) ---")
        print(content[:500])
        print("--- End of content preview ---")

        # Print all lines that might be related to accuracy
        print("\n--- Lines that might contain accuracy information ---")
        for line in content.splitlines():
            if any(term in line.lower() for term in ['acc', 'score', 'correct', 'precision', 'eval', 'test', 'valid', '%']):
                print(f"POTENTIAL LINE: {line}")
        print("--- End of potential accuracy lines ---\n")

        # Regex to find patterns like "accuracy: 73.2", "acc: 73.2", "accuracy is 73.2", "73.2%"
        # It captures the floating point number
        match = re.search(r'(?:accuracy|acc)[^0-9]*([0-9]+\\.?[0-9]*)', content, re.IGNORECASE)
        if match:
            accuracy = float(match.group(1))
            print(f"SUCCESS: Found accuracy (pattern 1): {accuracy}")
            return accuracy

        # If the above doesn't match, try to find a percentage value on a line with "accuracy" or "acc"
        match = re.search(r'([0-9]+\\.?[0-9]*)\\s*%', content, re.IGNORECASE)
        if match:
            line_with_percentage = ""
            for line in content.splitlines():
                if match.group(0) in line:
                    line_with_percentage = line
                    break
            if "accuracy" in line_with_percentage.lower() or "acc" in line_with_percentage.lower():
                accuracy = float(match.group(1))
                print(f"SUCCESS: Found accuracy (pattern 2 - percentage with keyword): {accuracy}%")
                return accuracy
        
        # Fallback for numbers like 0.732 if "accuracy" or "acc" is present
        match = re.search(r'(?:accuracy|acc)[^0-9]*([0]\\.[0-9]+)', content, re.IGNORECASE)
        if match:
            accuracy = float(match.group(1)) * 100 # Convert to percentage
            print(f"SUCCESS: Found accuracy (pattern 3 - decimal with keyword): {accuracy} (converted from {match.group(1)})")
            return accuracy

        # Special format for winogrande logs: "acc,none: 0.7387529597474349"
        match = re.search(r'acc,none:\s*([0-9]+\.?[0-9]*)', content, re.IGNORECASE)
        if match:
            accuracy = float(match.group(1)) * 100  # Convert to percentage
            print(f"SUCCESS: Found accuracy (pattern 5 - acc,none format): {accuracy}% (converted from {match.group(1)})")
            return accuracy

        # More generic percentage search if specific keywords are not found with it
        match = re.search(r'([0-9]+\\.?[0-9]*)\\s*%', content) # Case-sensitive for less noise if keyword missing
        if match:
            # This is a less specific match, so be cautious.
            # It might pick up other percentages if "accuracy" or "acc" are not nearby.
            # We could add a check here to see if "accuracy" or "acc" appears on the SAME line.
            potential_accuracy_line = ""
            for line in content.splitlines():
                if match.group(0) in line: # match.group(0) is the full percentage string e.g. "73.5%"
                    potential_accuracy_line = line.lower()
                    break
            
            # Check if "accuracy" or "acc" is on the same line as the percentage
            if "accuracy" in potential_accuracy_line or "acc" in potential_accuracy_line:
                 accuracy = float(match.group(1))
                 print(f"SUCCESS: Found accuracy (pattern 4 - generic percentage on line with keyword): {accuracy}% from line: '{potential_accuracy_line}'")
                 return accuracy
            else:
                print(f"INFO: Found a percentage {match.group(0)} but 'accuracy' or 'acc' not on the same line: '{potential_accuracy_line}'. This might not be the target accuracy.")


        print(f"Accuracy not found in the expected format in {log_file_path}.")
        return None
    except FileNotFoundError: # This should be caught by the os.path.exists check now
        print(f"ERROR: Log file not found at {log_file_path} (should have been caught earlier).")
        return None
    except Exception as e:
        print(f"ERROR: An error occurred while reading or parsing log file {log_file_path}: {e}")
        return None

if __name__ == '__main__':
    print("--- Debugging Accuracy Extraction ---")
    
    # Ensure the target log directory exists or the dummy file creation will fail
    log_dir = os.path.dirname(BASELINE_LOG_PATH)
    if not os.path.exists(log_dir):
        print(f"Log directory {log_dir} for baseline log does not exist. Creating it.")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create log directory {log_dir}: {e}. This might cause issues.")

    extracted_acc = extract_accuracy_from_log(BASELINE_LOG_PATH)
    
    if extracted_acc is not None:
        print(f"\\nFinal Extracted Accuracy: {extracted_acc}")
    else:
        print(f"\\nFailed to extract accuracy from {BASELINE_LOG_PATH}")
        print("Please check the log content and the regex patterns in this script.")
        print("If the script created a dummy file, replace its content with your actual log data and re-run.")

    # You can also test with a raw string for quick tests:
    # print("\\n--- Testing with a raw string ---")
    # RAW_LOG_CONTENT_FOR_TESTING = """
    # Some other lines
    # Test accuracy is 75.23%
    # More lines
    # """
    # # To test with raw string, you'd temporarily modify extract_accuracy_from_log
    # # to accept content directly, or write it to a temp file.
    # # For now, this is a placeholder to remind you of this testing technique.
    # temp_test_file = os.path.join(WORKSPACE_ROOT, 'tests', 'temp_log_for_debug.txt')
    # with open(temp_test_file, 'w') as f_temp:
    #     f_temp.write(RAW_LOG_CONTENT_FOR_TESTING)
    # extracted_acc_raw = extract_accuracy_from_log(temp_test_file)
    # if extracted_acc_raw is not None:
    #     print(f"Final Extracted Accuracy from RAW string test: {extracted_acc_raw}")
    # else:
    #     print(f"Failed to extract accuracy from RAW string test.")
    # os.remove(temp_test_file) # Clean up 