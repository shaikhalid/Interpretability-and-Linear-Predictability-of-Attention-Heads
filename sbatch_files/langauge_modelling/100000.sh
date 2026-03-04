DATASET="allenai/olmo-mix-1124"
TASK_TYPE="inference"
MODEL="allenai/OLMo-2-1124-7B"
MODEL_FAMILY="olmo2"
ALIAS="o7b424b"
REVISION="stage1-step101000-tokens424B"

export CUDA_VISIBLE_DEVICES=3
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 50 --revision $REVISION
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true