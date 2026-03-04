DATASET="winogrande"
TASK_TYPE="multiple_choice"
ONLY_DECODE=0

MODEL="tiiuae/Falcon3-10B-Instruct"
MODEL_FAMILY="llama3"
ALIAS="f10b"
export CUDA_VISIBLE_DEVICES=0
NUMBER_OF_SAMPLES=12000

# baseline
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 > sbatch_files/acc_vs_kv_prediction/logs/$(echo Baseline_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

# # # # #collection
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle k --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/acc_vs_kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/acc_vs_kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

# # # # #training
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true true > sbatch_files/acc_vs_kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

NUM_TARGET_HEADS=0.1
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${ONLY_DECODE} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 




NUM_TARGET_HEADS=0.2
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${ONLY_DECODE}  > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 


NUM_TARGET_HEADS=0.3
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${ONLY_DECODE}  > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 


NUM_TARGET_HEADS=0.4
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${ONLY_DECODE}  > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 


NUM_TARGET_HEADS=0.5
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${ONLY_DECODE}  > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 


NUM_TARGET_HEADS=0.6
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS}  > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 



NUM_TARGET_HEADS=0.7
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${ONLY_DECODE} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 



NUM_TARGET_HEADS=0.8
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${ONLY_DECODE} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 



NUM_TARGET_HEADS=0.9
python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_key.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json --output_plot_file gmls/thought_graph_key.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json

python -m key_prediction.find_ref_heads --graph_file_path gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_value.gml --output_json_path ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json --output_plot_file gmls/thought_graph_value.png --target_num_heads ${NUM_TARGET_HEADS} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Find_ref_heads_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.verify_heads ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json

# Train prediction models
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_key.py --file kv_pickles/k_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type k_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_keys.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_k_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1
python -m key_prediction.trainer --config key_prediction/config_linear_torch_gpu_value.py --file kv_pickles/v_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --alias $(echo ${DATASET}_${ALIAS} | tr '/' '_') --matrix_type v_matrix --head_selection_file ref_target_mapping/ref_target_heads_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_values.json > sbatch_files/acc_vs_kv_prediction/logs/$(echo Training_v_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1

# # # run eval with kv prediction
python main.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${ONLY_DECODE} > sbatch_files/acc_vs_kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS}_${NUM_TARGET_HEADS} | tr '/' '_').log 2>&1 