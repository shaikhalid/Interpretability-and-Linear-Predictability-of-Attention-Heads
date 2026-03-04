dataset=${1:-mmlu}
predict_keys=${2:-true}
predict_value=${3:-true}
preprocess=${4:-true}
thought_graph=${5:-true}

echo "Dataset: $dataset"
echo "Predict Keys: $predict_keys"
echo "Predict Value: $predict_value"
echo "Preprocess: $preprocess"
echo "Thought Graph: $thought_graph"

# Find reference and target heads
if [ "$predict_keys" = true ]; then
    python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_${dataset}_key.gml --output_json_path ref_target_mapping/ref_target_heads_${dataset}_keys.json --output_plot_file gmls/thought_graph_key.png 
    python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_${dataset}_keys.json
fi

if [ "$predict_value" = true ]; then
    python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_${dataset}_value.gml --output_json_path ref_target_mapping/ref_target_heads_${dataset}_values.json --output_plot_file gmls/thought_graph_value.png 
    python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_${dataset}_values.json
fi


# Train prediction models
if [ "$predict_keys" = true ]; then
    python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_${dataset}.pkl --alias ${dataset} --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_${dataset}_keys.json
fi

if [ "$predict_value" = true ]; then
    python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_${dataset}.pkl --alias ${dataset} --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_${dataset}_values.json
fi