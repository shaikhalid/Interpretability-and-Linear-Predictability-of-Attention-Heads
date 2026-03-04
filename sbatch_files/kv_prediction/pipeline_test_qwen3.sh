#!/bin/bash
# Record start time
overall_start_time=$(date +%s)
step_start_time=0
step_end_time=0
durations=()
step_names=()

DATASET="winogrande"
TASK_TYPE="multiple_choice"
DECODE_ONLY=0

MODEL="Qwen/Qwen3-32B"
MODEL_FAMILY="qwen3"
ALIAS="q32b"
export CUDA_VISIBLE_DEVICES=0
NUMBER_OF_SAMPLES=500

# # baseline
step_names+=("Baseline")
step_start_time=$(date +%s)
python main_flash.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 > sbatch_files/kv_prediction/logs/$(echo Baseline_${DATASET}_${ALIAS} | tr '/' '_').log
step_end_time=$(date +%s)
durations+=($((step_end_time - step_start_time)))

# # # #collection
step_names+=("Collection_k")
step_start_time=$(date +%s)
python main_flash.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle k --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log
step_end_time=$(date +%s)
durations+=($((step_end_time - step_start_time)))

step_names+=("Collection_v")
step_start_time=$(date +%s)
python main_flash.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log
step_end_time=$(date +%s)
durations+=($((step_end_time - step_start_time)))

# # #training
step_names+=("Preprocessing")
step_start_time=$(date +%s)
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true true > sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log
step_end_time=$(date +%s)
durations+=($((step_end_time - step_start_time)))

step_names+=("Training")
step_start_time=$(date +%s)
./run_training.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true true > sbatch_files/kv_prediction/logs/$(echo Training_${DATASET}_${ALIAS} | tr '/' '_').log
step_end_time=$(date +%s)
durations+=($((step_end_time - step_start_time)))

# # run eval with kv prediction
step_names+=("Prediction")
step_start_time=$(date +%s)
python main_flash.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size 8 --task_type ${TASK_TYPE}  --context_length 2048 --predict True --predict_keys True --predict_values True --alias ${ALIAS} --only_decode ${DECODE_ONLY} > sbatch_files/kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS} | tr '/' '_').log 
step_end_time=$(date +%s)
durations+=($((step_end_time - step_start_time)))

overall_end_time=$(date +%s)
total_duration=$((overall_end_time - overall_start_time))

echo "Execution Time Summary:"
for i in "${!durations[@]}"; do
  printf "Step %s: %s seconds\n" "${step_names[i]}" "${durations[i]}"
done
printf "Total execution time: %s seconds\n" "$total_duration"



