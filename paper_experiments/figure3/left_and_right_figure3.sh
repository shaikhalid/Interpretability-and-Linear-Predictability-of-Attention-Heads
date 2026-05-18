# # Define variables
DATASET="c4"
TASK_TYPE="inference"
LIMIT_SAMPLES=25
MATRIX_TO_PICKLE="v"
MATRIX_NAME="value"

# Map alias -> step label used by LAS_calc
step_dir_from_alias() {
  case "$1" in
    o7b1b)   echo "150"  ;;
    o7b5b)   echo "1k"   ;;
    o7b210b) echo "50k"  ;;
    o7b424b) echo "100k" ;;
    o7b1049b) echo "250k" ;;
    o7b3896b) echo "1M"   ;;
    *)       echo "$1"   ;;
  esac
}

copy_to_fig3() {
  local graph_path="$1"
  local alias="$2"
  local dir="$(step_dir_from_alias $alias)"
  local dest_dir="gmls/figure3_gmls/$dir"
  mkdir -p "$dest_dir"
  local dest_file="$dest_dir/$(basename "$graph_path")"
  cp "$graph_path" "$dest_file"
  if [ ! -f "$dest_file" ]; then
    echo "File not found"
    exit 1
  fi
}


MODEL="allenai/OLMo-2-1124-7B"
MODEL_FAMILY="olmo2"
ALIAS="o7b1b"
REVISION="stage1-step150-tokens1B"

# export CUDA_VISIBLE_DEVICES=0
# python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $LIMIT_SAMPLES --revision $REVISION
# echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
# ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true

# cp the output file to a new directory if doesn't exist
if [ ! -d "gmls/figure3_gmls" ]; then
    mkdir gmls/figure3_gmls
fi

GRAPH_PATH="gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_NAME}.gml"
if [ ! -f "$GRAPH_PATH" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $LIMIT_SAMPLES --revision $REVISION
    echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
    ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true
fi

copy_to_fig3 "$GRAPH_PATH" "$ALIAS"

ALIAS="o7b5b"
REVISION="stage1-step1000-tokens5B"
GRAPH_PATH="gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_NAME}.gml"
if [ ! -f "$GRAPH_PATH" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $LIMIT_SAMPLES --revision $REVISION
    echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
    ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true
fi

copy_to_fig3 "$GRAPH_PATH" "$ALIAS"

ALIAS="o7b210b"
REVISION="stage1-step50000-tokens210B"
GRAPH_PATH="gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_NAME}.gml"
if [ ! -f "$GRAPH_PATH" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $LIMIT_SAMPLES --revision $REVISION
    echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
    ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true
fi

copy_to_fig3 "$GRAPH_PATH" "$ALIAS"

ALIAS="o7b424b"
REVISION="stage1-step101000-tokens424B"
GRAPH_PATH="gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_NAME}.gml"
if [ ! -f "$GRAPH_PATH" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $LIMIT_SAMPLES --revision $REVISION
    echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
    ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true
fi

copy_to_fig3 "$GRAPH_PATH" "$ALIAS"


ALIAS="o7b1049b"
REVISION="stage1-step250000-tokens1049B"
GRAPH_PATH="gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_NAME}.gml"
if [ ! -f "$GRAPH_PATH" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $LIMIT_SAMPLES --revision $REVISION
    echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
    ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true
fi

copy_to_fig3 "$GRAPH_PATH" "$ALIAS"

ALIAS="o7b2098b"
REVISION="stage1-step500000-tokens2098B"
GRAPH_PATH="gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_NAME}.gml"
if [ ! -f "$GRAPH_PATH" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $LIMIT_SAMPLES --revision $REVISION
    echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
    ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true
fi

copy_to_fig3 "$GRAPH_PATH" "$ALIAS"


ALIAS="o7b3896b"
REVISION="stage1-step928646-tokens3896B"
GRAPH_PATH="gmls/thought_graph_$(echo ${DATASET}_${ALIAS} | tr '/' '_')_${MATRIX_NAME}.gml"
if [ ! -f "$GRAPH_PATH" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python main.py --model_type $MODEL_FAMILY --ckpt_dir $MODEL --few_shot_number 0 --dataset $DATASET --batch_size 1 --task_type $TASK_TYPE --context_length 2048 --kv_collection True --pickle_kv_cache True --matrix_to_pickle $MATRIX_TO_PICKLE --limit_samples $LIMIT_SAMPLES --revision $REVISION
    echo "Running prediction... $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true"
    ./run_preprocessing.sh $(echo ${DATASET}_${ALIAS} | tr '/' '_') false true
fi

copy_to_fig3 "$GRAPH_PATH" "$ALIAS"

python visualisation_helper/LAS_calc.py --input_dir gmls/figure3_gmls --output_dir plots/figure3