# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <task_type>"
    echo "Example: $0 gpqa_main_zeroshot multiple_choice"
    exit 1
fi

DATASET="$1"
TASK_TYPE="$2"
DECODE_ONLY=0

MODEL="tiiuae/Falcon3-10B-Instruct"
MODEL_FAMILY="llama3"
ALIAS="f10b"
export CUDA_VISIBLE_DEVICES=2
NUMBER_OF_SAMPLES=2000

# # baseline
# echo "Running baseline for ${DATASET} on Model: ${MODEL}"
# echo "Output: sbatch_files/kv_prediction/logs/$(echo Baseline_${DATASET}_${ALIAS} | tr '/' '_').log"
# python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 > sbatch_files/kv_prediction/logs/$(echo Baseline_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1


# # # # #collection
# echo "Running Key (K) states collection on ${NUMBER_OF_SAMPLES} samples (Tokens) for ${DATASET}"
# echo "Output: sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log"
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle k --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

# echo "Running Value (V) states collection on ${NUMBER_OF_SAMPLES} samples (Tokens) for ${DATASET}"
# echo "Output: sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log"
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1



# # # #training
# echo "Running preprocessing for collected K and V states"
# echo "Output: sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log"
# ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true true > sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

# echo "Running training LR classifier for 50% KV prediction"
# echo "Output: sbatch_files/kv_prediction/logs/$(echo Training_${DATASET}_${ALIAS} | tr '/' '_').log"
# ./run_training.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true true > sbatch_files/kv_prediction/logs/$(echo Training_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

# # delete the kv pickles
# rm -rf kv_pickles/k_cache_data_${DATASET}_${ALIAS}.pkl
# rm -rf kv_pickles/v_cache_data_${DATASET}_${ALIAS}.pkl

# rm -rf processed_data/key_${DATASET}_${ALIAS}.pkl
# rm -rf processed_data/value_${DATASET}_${ALIAS}.pkl


echo "Running evaluation with 50% KV prediction"
echo "Output: sbatch_files/kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS} | tr '/' '_').log"

python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${DECODE_ONLY} > sbatch_files/kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1


# Function to extract accuracy from log files
extract_accuracy_from_log() {
    local log_file_path="$1"
    
    if [ ! -f "$log_file_path" ]; then
        echo "Error: Log file not found at $log_file_path"
        return 1
    fi
    
    # Try different patterns to extract accuracy
    # Pattern 1: Special format for winogrande logs: "acc,none: 0.7387529597474349"
    local accuracy=$(grep -oiE 'acc,none:\s*([0-9]+\.[0-9]+)' "$log_file_path" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    if [ -n "$accuracy" ]; then
        # Convert decimal to percentage
        accuracy=$(echo "$accuracy * 100" | bc -l)
        printf "%.2f" "$accuracy"
        return 0
    fi
    
    # Pattern 2: Look for "accuracy: X.XX" or "acc: X.XX" patterns
    accuracy=$(grep -oiE '(accuracy|acc)[:, ]*([0-9]+\.?[0-9]*)' "$log_file_path" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)
    if [ -n "$accuracy" ]; then
        # Check if it's already a percentage (0-100) or a decimal (0-1)
        if (( $(echo "$accuracy < 1.0" | bc -l) )); then
            # Convert decimal to percentage
            accuracy=$(echo "$accuracy * 100" | bc -l)
        fi
        printf "%.2f" "$accuracy"
        return 0
    fi
    
    # Pattern 3: Look for percentage values with % symbol
    accuracy=$(grep -oE '[0-9]+\.?[0-9]*%' "$log_file_path" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)
    if [ -n "$accuracy" ]; then
        printf "%.2f" "$accuracy"
        return 0
    fi
    
    echo "N/A"
    return 1
}

echo ""
echo "=================================="
echo "        ACCURACY COMPARISON"
echo "=================================="

# Extract baseline accuracy
BASELINE_LOG="sbatch_files/kv_prediction/logs/$(echo Baseline_${DATASET}_${ALIAS} | tr '/' '_').log"
BASELINE_ACCURACY=$(extract_accuracy_from_log "$BASELINE_LOG")

echo "Baseline Accuracy: ${BASELINE_ACCURACY}%"

# Extract prediction accuracy
PREDICTION_LOG="sbatch_files/kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS} | tr '/' '_').log"
PREDICTION_ACCURACY=$(extract_accuracy_from_log "$PREDICTION_LOG")

echo "Accuracy with KV prediction: ${PREDICTION_ACCURACY}%"

# Calculate accuracy difference if both values are available
if [ "$BASELINE_ACCURACY" != "N/A" ] && [ "$PREDICTION_ACCURACY" != "N/A" ]; then
    ACCURACY_DIFF=$(echo "$PREDICTION_ACCURACY - $BASELINE_ACCURACY" | bc -l)
    echo "Accuracy Difference: $(printf "%.2f" "$ACCURACY_DIFF")% (KV prediction - Baseline)"
    
    if (( $(echo "$ACCURACY_DIFF >= 0" | bc -l) )); then
        echo "Status: Prediction accuracy is higher than or equal to baseline"
    else
        echo "Status: Prediction accuracy is lower than baseline"
    fi
else
    echo "Status: Could not calculate difference due to missing accuracy values"
fi

echo "=================================="



