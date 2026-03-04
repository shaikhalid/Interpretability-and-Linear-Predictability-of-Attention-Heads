DATASET="longbench_2wikimqa"
TASK_TYPE="longbench"

MODEL="tiiuae/Falcon3-10B-Instruct"
MODEL_FAMILY="llama3"
ALIAS="f10b"
export CUDA_VISIBLE_DEVICES=0
NUMBER_OF_SAMPLES=10
CONTEXT_LENGTH=70000
BATCH_SIZE=1

# baseline
python main_flash.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size ${BATCH_SIZE} --task_type ${TASK_TYPE}  --context_length ${CONTEXT_LENGTH} > sbatch_files/kv_prediction/logs/$(echo Baseline_${DATASET}_${ALIAS} | tr '/' '_').log

 # # #collection
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 10000 --kv_collection True --pickle_kv_cache True --matrix_to_pickle k --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True > sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 10000 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True > sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log

# # # #training
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true true > sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log
./run_training.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true true > sbatch_files/kv_prediction/logs/$(echo Training_${DATASET}_${ALIAS} | tr '/' '_').log

# # run eval with kv prediction
python main_flash.py --model_type ${MODEL_FAMILY} --ckpt_dir ${MODEL} --few_shot_number 0 --dataset ${DATASET} --batch_size ${BATCH_SIZE} --task_type ${TASK_TYPE}  --context_length ${CONTEXT_LENGTH} --predict True --predict_keys True --predict_values True --alias ${ALIAS}  > sbatch_files/kv_prediction/logs/$(echo Prediction_${DATASET}_${ALIAS} | tr '/' '_').log 



