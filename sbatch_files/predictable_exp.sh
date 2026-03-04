# # Define variables
# DATASET="predictable_sequence"
# TASK_TYPE="inference"

# # 1st checkpoint
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_FAMILY="llama3"
# ALIAS="l8b"

# export CUDA_VISIBLE_DEVICES=0
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 50 --alias $ALIAS
# ./run_prediction.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true

# DATASET="random_sequence"
# TASK_TYPE="inference"

# # 2nd checkpointq
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_FAMILY="llama3"
# ALIAS="l8b"

# export CUDA_VISIBLE_DEVICES=0
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 50 --alias $ALIAS
# ./run_prediction.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true



# DATASET="allenai/c4"
# TASK_TYPE="inference"

# # 1st checkpoint
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_FAMILY="llama3"
# ALIAS="l8b"

# export CUDA_VISIBLE_DEVICES=2
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 100 --alias $ALIAS --apply_chat_template True
# echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
# ./run_prediction.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true

# rm -rf kv_pickles/*.pkl
# rm -rf processed_data/*.pkl


DATASET="random_words"
TASK_TYPE="inference"

# 1st checkpoint
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_FAMILY="llama3"
ALIAS="l8b"

export CUDA_VISIBLE_DEVICES=3
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 25 --alias $ALIAS --apply_chat_template True --context_length 256
echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true



