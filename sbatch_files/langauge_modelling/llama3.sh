# # Define variables
DATASET="gsm8k"
TASK_TYPE="math"
MODEL="tiiuae/Falcon3-10B-Instruct"
MODEL_FAMILY="llama3"
ALIAS="f10b_graph"
MATRIX_TO_PICKLE="v"
NUMBER_OF_SAMPLES=50   

export CUDA_VISIBLE_DEVICES=3
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS


python -m key_prediction.preprocess_activations --input kv_pickles/${MATRIX_TO_PICKLE}_cache_data_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --matrix_type ${MATRIX_TO_PICKLE}_matrix --output processed_data/${MATRIX_TO_PICKLE}_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl
python -m key_prediction.trace_thought --data_file_path processed_data/${MATRIX_TO_PICKLE}_$(echo ${DATASET}_${ALIAS} | tr '/' '_').pkl --output_graph_file gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_TO_PICKLE}.gml --output_plot_file gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_TO_PICKLE}.png --num_samples 15000

