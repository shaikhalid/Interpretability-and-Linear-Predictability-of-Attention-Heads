# ./sbatch_files/kv_prediction/llama3_kv_model.sh c4 language_modeling

DATASET="c4"
TASK_TYPE="language_modeling"
DECODE_ONLY=0

MODEL="tiiuae/Falcon3-10B-Instruct"
MODEL_FAMILY="llama3"
ALIAS="f10b"
export CUDA_VISIBLE_DEVICES=0
NUMBER_OF_SAMPLES=2000

# # #collection
echo "Running Key (K) states collection on ${NUMBER_OF_SAMPLES} samples for ${DATASET}"
echo "Output: sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log"
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle k --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_k_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

echo "Running Value (V) states collection on ${NUMBER_OF_SAMPLES} samples for ${DATASET}"
echo "Output: sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log"
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle v --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_v_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

echo "Running Query (Q) states collection on ${NUMBER_OF_SAMPLES} samples for ${DATASET}"
echo "Output: sbatch_files/kv_prediction/logs/$(echo Collection_q_${DATASET}_${ALIAS} | tr '/' '_').log"
python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle q --limit_samples $NUMBER_OF_SAMPLES --alias $ALIAS --apply_chat_template True --random_sampling 1 > sbatch_files/kv_prediction/logs/$(echo Collection_q_${DATASET}_${ALIAS} | tr '/' '_').log 2>&1

# # # preprocessing
echo "Running preprocessing for collected K, V, and Q states"
echo "Output: sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS} | tr '/' '_').log"

python -m key_prediction.preprocess_activations --input kv_pickles/k_cache_data_${DATASET}_${ALIAS}.pkl --matrix_type k_matrix --output processed_data/key_${DATASET}_${ALIAS}.pkl > sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS}_key | tr '/' '_').log 2>&1
python -m key_prediction.preprocess_activations --input kv_pickles/v_cache_data_${DATASET}_${ALIAS}.pkl --matrix_type v_matrix --output processed_data/value_${DATASET}_${ALIAS}.pkl > sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS}_value | tr '/' '_').log 2>&1
python -m key_prediction.preprocess_activations --input kv_pickles/q_cache_data_${DATASET}_${ALIAS}.pkl --matrix_type q_matrix --output processed_data/query_${DATASET}_${ALIAS}.pkl > sbatch_files/kv_prediction/logs/$(echo Preprocessing_${DATASET}_${ALIAS}_query | tr '/' '_').log 2>&1

python -m key_prediction.trace_thought --data_file_path processed_data/key_${DATASET}_${ALIAS}.pkl --output_graph_file gmls/thought_graph_${DATASET}_${ALIAS}_key.gml --output_plot_file gmls/thought_graph_${DATASET}_${ALIAS}_key.png --num_samples 10000 > sbatch_files/kv_prediction/logs/$(echo Trace_thought_${DATASET}_${ALIAS}_key | tr '/' '_').log 2>&1
python -m key_prediction.trace_thought --data_file_path processed_data/value_${DATASET}_${ALIAS}.pkl --output_graph_file gmls/thought_graph_${DATASET}_${ALIAS}_value.gml --output_plot_file gmls/thought_graph_${DATASET}_${ALIAS}_value.png --num_samples 10000 > sbatch_files/kv_prediction/logs/$(echo Trace_thought_${DATASET}_${ALIAS}_value | tr '/' '_').log 2>&1
python -m key_prediction.trace_thought --data_file_path processed_data/query_${DATASET}_${ALIAS}.pkl --output_graph_file gmls/thought_graph_${DATASET}_${ALIAS}_query.gml --output_plot_file gmls/thought_graph_${DATASET}_${ALIAS}_query.png --num_samples 10000 > sbatch_files/kv_prediction/logs/$(echo Trace_thought_${DATASET}_${ALIAS}_query | tr '/' '_').log 2>&1

#avg r2 across N for K, V, and Q
echo "Running avg R2 analysis across N for K, V, and Q components"
echo "Output: sbatch_files/kv_prediction/logs/$(echo AvgR2_${DATASET}_${ALIAS} | tr '/' '_').log"
python visualisation_helper/avg_r2_across_N_kqv.py \
  --k_gml gmls/thought_graph_${DATASET}_${ALIAS}_key.gml \
  --q_gml gmls/thought_graph_${DATASET}_${ALIAS}_query.gml \
  --v_gml gmls/thought_graph_${DATASET}_${ALIAS}_value.gml \
  --k_pickle processed_data/key_${DATASET}_${ALIAS}.pkl \
  --q_pickle processed_data/query_${DATASET}_${ALIAS}.pkl \
  --v_pickle processed_data/value_${DATASET}_${ALIAS}.pkl \
  --max_n 5 \
  --num_samples_subsample 2000 

rm  -rf processed_data/key_${DATASET}_${ALIAS}.pkl
rm  -rf processed_data/value_${DATASET}_${ALIAS}.pkl
rm  -rf processed_data/query_${DATASET}_${ALIAS}.pkl
rm  -rf kv_pickles/k_cache_data_${DATASET}_${ALIAS}.pkl
rm  -rf kv_pickles/v_cache_data_${DATASET}_${ALIAS}.pkl
rm  -rf kv_pickles/q_cache_data_${DATASET}_${ALIAS}.pkl