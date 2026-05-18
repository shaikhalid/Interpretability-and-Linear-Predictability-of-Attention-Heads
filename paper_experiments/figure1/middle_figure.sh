# ./sbatch_files/kv_prediction/llama3_kv_model.sh c4 language_modeling

DATASET="c4"
TASK_TYPE="language_modeling"
DECODE_ONLY=0

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_FAMILY="llama3"
ALIAS="l8b"
export CUDA_VISIBLE_DEVICES=0
NUMBER_OF_SAMPLES=2000

# # # #collection
echo "Running Key (K) states collection on ${NUMBER_OF_SAMPLES} samples for ${DATASET}"
echo "Output: sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log"
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle k --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

echo "Running Value (V) states collection on ${NUMBER_OF_SAMPLES} samples for ${DATASET}"
echo "Output: sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log"
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

# # #preprocessing
echo "Running preprocessing for collected K and V states"
echo "Output: sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log"
./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') true true > sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

python visualisation_helper/recompute_r2_from_top_n.py gmls/thought_graph_c4_l8b_key.gml processed_data/key_c4_l8b.pkl  --matrix_key k_matrix --max_n 5 -o plots/experiment1_top5 --title_suffix ""

rm  -rf processed_data/key_c4_l8b.pkl
rm  -rf processed_data/value_c4_l8b.pkl
rm  -rf kv_pickles/k_cache_data_c4_l8b.pkl
rm  -rf kv_pickles/v_cache_data_c4_l8b.pkl