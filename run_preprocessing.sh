#!/bin/bash

# Usage: ./run_prediction.sh <dataset> <predict_keys> <predict_value> [skip_preprocess]
# Example: ./run_prediction.sh mmlu_college_biology true false

# Set default values if not provided
# dataset: mmlu, predict_keys: true, predict_value: false, skip_preprocess: false

# ./run_prediction.sh mmlu_college_biology true false false false
dataset_alias=${1:-c4_l8b}
predict_keys=${2:-true}
predict_value=${3:-true}
preprocess=${4:-true}
thought_graph=${5:-true}

echo "Dataset and Model Alias: $dataset_alias"
echo "Predict Keys: $predict_keys"
echo "Predict Value: $predict_value"
echo "Preprocess: $preprocess"
echo "Thought Graph: $thought_graph"


# if predict_keys is true, preprocess keys
if [ "$preprocess" = true ]; then
    echo "Preprocessing data..."

    if [ "$predict_keys" = true ]; then
        python -m key_prediction.preprocess_activations --input kv_pickles/k_cache_data_${dataset_alias}.pkl --matrix_type k_matrix --output processed_data/key_${dataset_alias}.pkl
    fi

    # if predict_value is true, preprocess values
    if [ "$predict_value" = true ]; then
        python -m key_prediction.preprocess_activations --input kv_pickles/v_cache_data_${dataset_alias}.pkl --matrix_type v_matrix --output processed_data/value_${dataset_alias}.pkl
    fi
fi
# ask user permission to proceed
# read -p "Data preprocessing complete. Press any key to continue..."

# Make the thought graph
if [ "$thought_graph" = true ]; then
    if [ "$predict_keys" = true ]; then
        python -m key_prediction.trace_thought --data_file_path processed_data/key_${dataset_alias}.pkl --output_graph_file gmls/thought_graph_${dataset_alias}_key.gml --output_plot_file gmls/thought_graph_${dataset_alias}_key.png --num_samples 25000
    fi

    if [ "$predict_value" = true ]; then
        python -m key_prediction.trace_thought --data_file_path processed_data/value_${dataset_alias}.pkl --output_graph_file gmls/thought_graph_${dataset_alias}_value.gml --output_plot_file gmls/thought_graph_${dataset_alias}_value.png --num_samples 25000
    fi
fi

