# # Define variables
DATASET="allenai/olmo-mix-1124"
TASK_TYPE="inference"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_FAMILY="random_weights_llama3"
ALIAS="l8b_random"

export CUDA_VISIBLE_DEVICES=3
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 50 --alias $ALIAS
echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true

