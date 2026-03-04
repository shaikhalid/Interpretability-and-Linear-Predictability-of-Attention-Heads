# # Define variables
DATASET="gsm8k"
TASK_TYPE="math"

# 1st checkpoint
# MODEL="allenai/OLMo-2-1124-7B"
# MODEL_FAMILY="olmo2"
# ALIAS="o7b42b"
# revision="stage1-step10000-tokens42B"

# export CUDA_VISIBLE_DEVICES=0
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 100 --revision $revision
# mv kv_cache_data.pkl kv_pickles/v_cache_data_${DATASET}_${ALIAS}.pkl
# ./run_prediction.sh ${DATASET}_${ALIAS} false true

# 2nd checkpoint
# MODEL="allenai/OLMo-2-1124-7B"
# MODEL_FAMILY="olmo2"
# ALIAS="o7b500b"
# REVISION="stage1-step119000-tokens500B"

# export CUDA_VISIBLE_DEVICES=1
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 100 --revision $REVISION
# mv kv_cache_data.pkl kv_pickles/v_cache_data_${DATASET}_${ALIAS}.pkl
# ./run_prediction.sh ${DATASET}_${ALIAS} false true

# 3rd checkpoint
MODEL="allenai/OLMo-2-1124-7B"
MODEL_FAMILY="olmo2"
ALIAS="o7b2513b"
revision="stage1-step599000-tokens2513B"

export CUDA_VISIBLE_DEVICES=2
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 100 --revision $revision
mv kv_cache_data.pkl kv_pickles/v_cache_data_${DATASET}_${ALIAS}.pkl
./run_prediction.sh ${DATASET}_${ALIAS} false true

# 4th checkpoint
# MODEL="allenai/OLMo-2-1124-7B"
# MODEL_FAMILY="olmo2"
# ALIAS="o7b3896b"
# REVISION="stage1-step928646-tokens3896B"

# export CUDA_VISIBLE_DEVICES=3
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 100 --revision $REVISION
# mv kv_cache_data.pkl kv_pickles/v_cache_data_${DATASET}_${ALIAS}.pkl
# ./run_prediction.sh ${DATASET}_${ALIAS} false true

# MODEL="allenai/OLMo-2-1124-7B"
# MODEL_FAMILY="olmo2"
# ALIAS="o7b"
# revision="main"

# export CUDA_VISIBLE_DEVICES=4
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 100 --revision $revision
# mv kv_cache_data.pkl kv_pickles/v_cache_data_${DATASET}_${ALIAS}.pkl
# ./run_prediction.sh humaneval_o7b false true