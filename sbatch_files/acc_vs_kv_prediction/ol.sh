

DATASET="gsm8k"
TASK_TYPE="math"

MODEL="allenai/OLMo-2-0425-1B-Instruct"
MODEL_FAMILY="olmo2"
ALIAS="o1b"
NUM_TARGET_HEADS=0.5
export CUDA_VISIBLE_DEVICES=1

# # baseline
# python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 > sbatch_files/kv_prediction/logs/$(echo Baseline_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # # # #collection
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle k --limit_samples 50 --alias $ALIAS --apply_chat_template True --dtype float32 > sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 50 --alias $ALIAS --apply_chat_template True > sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log

# # # # #training
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true false > sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS}
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

# python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS}  
# python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json
# python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS}  > sbatch_files/kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 