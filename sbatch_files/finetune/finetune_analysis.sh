#!/bin/bash

DATASET="gsm8k"
TASK_TYPE="math"
MODEL="tiiuae/Falcon3-7B-Base"
MODEL_FAMILY="llama3"
ALIAS="f7b"

export CUDA_VISIBLE_DEVICES=2
# baseline eval
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 8 --task_type $TASK_TYPE --context_length 2048 > sbatch_files/finetune/logs/$(echo Baseline_${DATASET}_${ALIAS} | tr '/' '_').log

# # # basline kv collection
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples 100 --alias $ALIAS --context_length 512 > sbatch_files/finetune/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log

# # preprocess baseline
# ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true > sbatch_files/finetune/logs/$(echo Preprocessing_Baseline_${DATASET}_${ALIAS} | tr '/' '_').log

ALIAS="f7b_ft"
# finetune and eval and kv collection
python finetune.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --batch_size 8 --context_length 512 --dataset $DATASET --target_layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 --save_kv_cache True --matrix_to_pickle v --alias $ALIAS --lr 2e-4 --epochs 5 > sbatch_files/finetune/logs/$(echo Finetune_${DATASET}_${ALIAS} | tr '/' '_').log

# preprocess finetuned model
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true > sbatch_files/finetune/logs/$(echo Preprocessing_Finetuned_${DATASET}_${ALIAS} | tr '/' '_').log
