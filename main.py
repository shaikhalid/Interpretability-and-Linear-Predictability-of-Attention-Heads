import argparse
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from datasets import load_dataset, load_from_disk
from models.modelling_llama3_viz import (
    LlamaForCausalLM as Llama3ForCausalLM,
    LlamaAttention as Llama3Attention,
    LlamaFlashAttention2 as Llama3FlashAttention,
)
from models.modelling_qwen3 import Qwen3ForCausalLM, Qwen3Attention
from models.modelling_olmo2 import Olmo2ForCausalLM, Olmo2Attention
from models.qwen3_modelling_aug_collect import *
from models.olmo2_modelling_aug_collect import *
from models.llama3_modelling_aug_collect import *
from models.llama3_modelling_aug_predict import llama3_atten_aug_forward_predict
from models.qwen3_modelling_aug_predict import qwen3_atten_aug_forward_predict
from models.olmo2_modelling_aug_predict import olmo2_atten_aug_forward_predict
from models.llama3_modelling_aug_change_focus import llama3_atten_aug_forward_change_focus
from models.calibrate import Calibrate
from utils.data_utils import *
import os
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

choices = ["A", "B", "C", "D", "E"]

models_to_layers = {
    "llama2": 32,
    "llama3": 32,
    "llama3-8b-instruct": 32,
    "llama": 32,
    "mistral": 32,
    "gemma2-2b": 26,
    "llama3-3b": 32,
    "gemma2": 26
}

revision_to_alias ={
    "stage1-step10000-tokens42B": "o7b42b",
    "stage1-step119000-tokens500B": "o7b500b",
    "stage1-step599000-tokens2513B": "o7b2513b",
    "stage1-step928646-tokens3896B": "o7b3896b",
    "stage1-step150-tokens1B": "o7b1b",
    "stage1-step1000-tokens5B": "o7b5b",
    "stage1-step101000-tokens424B": "o7b424b",
    "stage1-step250000-tokens1049B": "o7b1049b",
    "stage1-step50000-tokens210B": "o7b210b",
    "stage1-step500000-tokens2098B": "o7b2098b",
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load(ckpt_dir, model_type, dtype):
    # Use model-specific tokenizer for different model types
    
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=False,
        cache_dir="../.cache/",
        padding_side="left",
    )
    
    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    if "llama" in model_type:
        tokenizer.bos_token_id = 1

    Calibrate.tokenizer = tokenizer


    model_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32

    if "random_weights_llama3" == model_type:
        # 1. Load the model's configuration
        print(f"Loading configuration for {ckpt_dir}...")
        config = AutoConfig.from_pretrained(ckpt_dir)
        print(config)
        config._attn_implementation = "eager"
        config.cache_dir = "../.cache"

        # 2. Instantiate the model from the configuration
        # This will initialize the model with random weights.
        print("Initializing the model with random weights using the loaded configuration...")
        model = Llama3ForCausalLM(config)
        model.to("cuda")

        first_layer_weights = model.model.layers[0].self_attn.q_proj.weight
        print(f"\nSample weights from the first attention projection layer (q_proj):")
        print(f"  Mean: {first_layer_weights.mean().item()}")
        print(f"  Std:  {first_layer_weights.std().item()}")

    elif "random_weights_olmo2" == model_type:
        set_seed(1647)
        print("Initializing the model with random weights...")
        config = AutoConfig.from_pretrained(ckpt_dir)
        print(config)
        config._attn_implementation = "eager"
        config.cache_dir = "../.cache"
        print("Initializing the model with random weights...")
        model = Olmo2ForCausalLM(config)
        model.to("cuda")

        first_layer_weights = model.model.layers[0].self_attn.q_proj.weight
        print(f"\nSample weights from the first attention projection layer (q_proj):")
        print(f"  Mean: {first_layer_weights.mean().item()}")
        print(f"  Std:  {first_layer_weights.std().item()}")
    
    elif "llama" in model_type:
        model = Llama3ForCausalLM.from_pretrained(
            ckpt_dir,
            low_cpu_mem_usage=True,
            cache_dir="../.cache",
            torch_dtype=model_dtype,
            attn_implementation="eager",
        )
        model.to("cuda")

    elif "olmo2" in model_type:
        model = Olmo2ForCausalLM.from_pretrained(
            ckpt_dir,
            revision=args.revision,
            low_cpu_mem_usage=True,
            cache_dir="../.cache",
            torch_dtype=model_dtype,
        )
        model.to("cuda")
    
    elif "qwen3" in model_type.lower():
        model = Qwen3ForCausalLM.from_pretrained(
            ckpt_dir,
            low_cpu_mem_usage=True,
            cache_dir="../.cache",
            torch_dtype=model_dtype,
        )
        model.to("cuda")

    else:
        raise ValueError(f"Model type {model_type} not supported")
        
    print(model)
    # if the model has chat template then print the chat template
    # Check if the model has a chat template and print it
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        print("\nModel chat template:")
        print(tokenizer.chat_template)
    return model, tokenizer

def collect_activations(model, tokenizer, tasks=None, batch_size=8):
    """
    Collect activations from the model
    """
    task = tasks[0]
    print(f"Collecting activations for task: {task}")

    # load the dataset using huggingface
    if args.task_type == "longbench":
        evaluate_with_lm_harness(model, tokenizer, tasks=[task], batch_size=args.batch_size)
        return
    
    if args.task_type == "ruler":
        evaluate_with_lm_harness(model, tokenizer, tasks=[task], batch_size=args.batch_size)
        return
    
    if task == "wikitext" or task == "c4" or task == "ifeval" or task == "truthfulqa_mc1":
        evaluate_with_lm_harness(model, tokenizer, tasks=[task], batch_size=args.batch_size)
        return
    
    elif task == "eli5":
        dataset = load_dataset("sentence-transformers/eli5", "pair", split="train", streaming=True, cache_dir="../.cache").shuffle(seed=42).take(args.limit_samples)
    
    elif task == "winogrande":
        dataset = load_dataset("winogrande", "winogrande_debiased", split="train", streaming=True, cache_dir="../.cache").shuffle(seed=42).take(args.limit_samples)
    
    elif task == "hellaswag":
        dataset = load_dataset("hellaswag", "hellaswag", split="train", streaming=True, cache_dir="../.cache").shuffle(seed=42).take(args.limit_samples)

    elif task == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="train", streaming=True, cache_dir="../.cache").shuffle(seed=42).take(args.limit_samples)
    
    elif task == "mmlu_stem":
        dataset = load_dataset("TIGER-Lab/MMLU-STEM", "default", split="test", streaming=True, cache_dir="../.cache").shuffle(seed=42).take(args.limit_samples)
    
    elif task == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", streaming=True, cache_dir="../.cache").shuffle(seed=42).take(args.limit_samples)
    
    else:
        raise ValueError(f"Dataset {task} not supported")
    
    # tokenize the dataset
    for sample in dataset:
       
        # generate outputs
        if "winogrande" == task:
            text = sample['sentence'] + " Option 1: " + sample['option1'] + " Option 2: " + sample['option2'] + " Answer: " + sample['answer']
        
        elif "hellaswag" == task:
            text = sample["ctx"] + " " + " ".join([ending for ending in sample["endings"]]) + " Answer: " + sample["label"]
        
        elif "gsm8k" == task:
            # prompt as per gsm8k_cot_llama
            text = " Given the following problem, reason and give a final answer to the problem.\nProblem: {{question}}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n" + sample["question"] + " The final answer is " + sample["answer"]

        elif "mmlu_stem" == task:
            text = sample["question"] + " " + " ".join([f"{i} {sample["choices"][i-1]}" for i in range(1, 5)])+ " Answer: " + str(sample["answer"])

        elif "gpqa" == task:
            text = sample["Question"] + " " + sample["Correct Answer"]

        elif "eli5" == task:
            text = sample["question"] + " " + sample["answer"]

        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=2, num_beams=1)

def inference_with_huggingface(model, tokenizer, tasks=None, batch_size=8):
    """
    Evaluate the model using HuggingFace's evaluate function
    """
    task = tasks[0]
    print(f"Evaluating model on tasks: {task}")

    # load the dataset using huggingface

    print(f"Loading dataset: {task}")
    if task == "allenai/olmo-mix-1124":
        dataset = load_dataset("allenai/olmo-mix-1124", "pes2o", split="train", streaming=True, cache_dir="../.cache").shuffle(seed=42).take(args.limit_samples) 
    elif task == "allenai/c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, cache_dir="../.cache").shuffle(seed=42).take(args.limit_samples)
    elif task == "predictable_sequence":
        dataset = load_from_disk("./datasets/predictable_sequence")
    elif task == "random_sequence":
        dataset = load_from_disk("./datasets/random_sequence")
    elif task == "random_words":
        dataset = load_from_disk("./datasets/random_words")
    else:
        raise ValueError(f"Dataset {task} not supported")
   
    # Process the dataset in batches
    for sample in dataset:
        # Tokenize the batch of texts
        if task == "predictable_sequence":
            text = ' '.join(sample['tokens'])
        elif task == "random_sequence":
            text = ' '.join(sample['tokens'])
        elif task == "random_words":
            text = ' '.join(sample['tokens'][:args.context_length])
        else:
            text = " ".join(sample['text'].split(" ")[:args.context_length])

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate outputs
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            num_beams=1,
        )

def evaluate_with_lm_harness(model, tokenizer, tasks=None, batch_size=8):
    """
    Evaluate the model using EleutherAI's lm-evaluation-harness
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        tasks: List of tasks to evaluate on. If None, uses a default set.
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary of evaluation results
    """

    
    print(f"Evaluating model on tasks: {tasks}")
    model.eval()

   
    # Define the HuggingFace model adapter
    eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device_map="auto",
        batch_size=batch_size,
        max_length=args.context_length,
    )
    # Run the evaluation
    print("Starting evaluation with lm-evaluation-harness...")

    if args.model_type == "olmo2":
        apply_chat_template = False
    else:
        apply_chat_template = args.apply_chat_template

    if args.dataset == "gsm8k":
        results = evaluator.simple_evaluate(
            model=eval_model,
            tasks=tasks,
            num_fewshot=args.few_shot_number,
            limit=args.limit_samples,
            apply_chat_template=apply_chat_template,
            metadata={'pretrained': args.ckpt_dir}
        )
    elif args.dataset == "humaneval":
        results = evaluator.simple_evaluate(
            model=eval_model,
            tasks=tasks,
            num_fewshot=0,
            confirm_run_unsafe_code=True,
            limit=args.limit_samples,
            apply_chat_template=args.apply_chat_template,
            metadata={'pretrained': args.ckpt_dir}
        )
    elif args.dataset == "gpqa":
        results = evaluator.simple_evaluate(
            model=eval_model,
            tasks=["gpqa_main_zeroshot"],
            num_fewshot=0,
            limit=args.limit_samples,
            apply_chat_template=args.apply_chat_template,
            metadata={'pretrained': args.ckpt_dir}
        )
    else:
        results = evaluator.simple_evaluate(
            model=eval_model,
            tasks=tasks,
            num_fewshot=args.few_shot_number,
            limit=args.limit_samples,
            apply_chat_template=args.apply_chat_template,
            metadata={'pretrained': args.ckpt_dir},
            log_samples=True
        )
        print(results["samples"])  

    # Extract and format results
    formatted_results = {}
    for task_name, task_results in results["results"].items():
        print(f"Results for {task_name}:")
        for metric, value in task_results.items():
            print("  " + str(metric) + ": " + str(value))
            formatted_results["{}_{}".format(task_name, metric)] = value
    
    # Print overall average if available
    if "averaged_results" in results:
        print("\nAveraged results:")
        for metric, value in results["averaged_results"].items():
            print("  " + str(metric) + ": " + str(value))
            formatted_results["average_{}".format(metric)] = value
    
    return formatted_results


    
def main(args):
    set_random_seed(42)
    # configure
    if args.matrix_to_pickle == "q":
        Calibrate.matrix_to_pickle = "q"
    elif args.matrix_to_pickle == "k":
        Calibrate.matrix_to_pickle = "k"
    elif args.matrix_to_pickle == "v":
        Calibrate.matrix_to_pickle = "v"
    
    Calibrate.default_prediction_model = args.default_prediction_model

    if args.apply_chat_template or (args.dataset == "gsm8k" and args.model_type != "olmo2"):
        Calibrate.apply_chat_template = True
        print("============ Using chat template ============")  
    else:
        Calibrate.apply_chat_template = False

    if args.only_decode == 1:
        Calibrate.only_decode = True
    else:
        Calibrate.only_decode = False

    model, tokenizer = load(args.ckpt_dir, args.model_type, args.dtype)
    Calibrate.dataset = args.dataset.replace("/", "_")
    Calibrate.model_alias = revision_to_alias[args.revision] if args.revision is not None else args.alias
    Calibrate.model = args.ckpt_dir
    Calibrate.limit_samples = args.limit_samples
    Calibrate.token_limit = args.token_limit
    Calibrate.random_sampling = args.random_sampling
    if args.kv_collection:
        print("============ Using layer-wise KV collection ============")  
        # Check if KV cache data file already exists and delete it if it does    
        
        
        kv_cache_file =  f'kv_pickles/{args.matrix_to_pickle}_cache_data_{Calibrate.dataset}_{Calibrate.model_alias}.pkl'

        if os.path.exists(kv_cache_file):
            print(f"Removing existing KV cache file: {kv_cache_file}")
            os.remove(kv_cache_file)
        print("============ Modifying model attention forward function ============")  
        Llama3Attention.forward = llama3_atten_aug_forward_collect
        Qwen3Attention.forward = qwen3_atten_aug_forward_collect
        Olmo2Attention.forward = olmo2_atten_aug_forward_collect

        Calibrate.set_pickle_kv_cache(True)
        
        collect_activations(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
        return


    elif args.predict:
        if args.predict_keys or args.predict_values:
            if args.predict_keys:
                Calibrate.predict_keys = True
            if args.predict_values:
                Calibrate.predict_values = True
            print("============ Using trained model weights for prediction ============")  
            Llama3Attention.forward = llama3_atten_aug_forward_predict
            # Llama3FlashAttention.forward = llama3_flash_atten_aug_forward_predict
            Qwen3Attention.forward = qwen3_atten_aug_forward_predict
            Olmo2Attention.forward = olmo2_atten_aug_forward_predict
        else:
            raise ValueError("At least one of predict_keys or predict_values must be True.")
        
    elif args.change_focus_exp:
        print("============ Using change focus experiment ============")  
        Llama3Attention.forward = llama3_atten_aug_forward_change_focus
        
        Calibrate.change_focus = args.change_focus
        Calibrate.focus_layer = args.focus_layer
        Calibrate.focus_head = args.focus_head
        Calibrate.reference_layer = args.reference_layer
        Calibrate.reference_head = args.reference_head
    else:
        print("============ Using original model ============")  

    
    if args.task_type == "multiple_choice":
        if "mmlu" in args.dataset:
            acc = evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
        elif "hellaswag" in args.dataset:
            acc = evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
        elif "truthfulqa_mc1" in args.dataset:
            acc = evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
        else:
            acc = evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
    
    elif args.task_type == "math":
        if args.dataset == "gsm8k":
            if "llama" in args.model_type or "mistral" in args.model_type:
                acc = evaluate_with_lm_harness(model, tokenizer, tasks=['gsm8k_cot_llama'], batch_size=args.batch_size)
            else:
                acc = evaluate_with_lm_harness(model, tokenizer, tasks=['gsm8k_cot_llama'], batch_size=args.batch_size)
        else:
            acc = evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
    
    elif args.task_type == "coding":
        acc = evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
            
    elif args.task_type == "language_modelling":
        if args.dataset == "wikitext":
            acc = evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
            print(f"Perplexity: {acc}")
    
    elif args.task_type == "inference":
        inference_with_huggingface(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
    
    else:
        evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=args.batch_size)
    # After evaluation
    if args.change_focus_exp:
        
        for layer_head in Calibrate.focus_layer_sink_tokens:
            print(f"Layer {layer_head} sink tokens: {Calibrate.focus_layer_sink_tokens[layer_head]}")

        for layer_head in Calibrate.ref_layer_sink_tokens:
            print(f"Layer {layer_head} sink tokens: {sorted(list(Calibrate.ref_layer_sink_tokens[layer_head]))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--few_shot_number", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--task_type", type=str, default="multiple_choice")
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--limit_samples", type=int, default=None,
                       help="Limit the number of samples to run KV cache analysis on")
    parser.add_argument("--pickle_kv_cache", type=bool, default=False,
                       help="Pickle the KV cache")
    parser.add_argument("--kv_collection", type=bool, default=False,
                       help="Enable KV collection")
    parser.add_argument("--predict", type=bool, default=False,
                       help="Enable prediction")
    parser.add_argument("--predict_keys", type=bool, default=False,)
    parser.add_argument("--predict_values", type=bool, default=False)

    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                       help="Data type for model loading (e.g., float16, bfloat16, float32)")
    parser.add_argument("--matrix_to_pickle", type=str, default="k", choices=["q", "k", "v"],
                       help="Which matrix to pickle when kv_collection is enabled (q, k, or v)")
    parser.add_argument("--change_focus_exp", type=bool, default=False,
                          help="Enable change focus experiment")
    parser.add_argument("--change_focus", type=bool, default=False)
    parser.add_argument("--focus_layer", type=int, default=-1)
    parser.add_argument("--focus_head", type=int, default=0)
    parser.add_argument("--reference_layer", type=int, default=0)
    parser.add_argument("--reference_head", type=int, default=0)
    parser.add_argument("--apply_chat_template", type=bool, default=False)
    parser.add_argument("--only_decode", type=int, default=0)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--alias", type=str, default=None)
    parser.add_argument("--random_sampling", type=int, default=0)
    parser.add_argument("--default_prediction_model", type=int, default=0)
    parser.add_argument("--token_limit", type=int, default=30000)
    args = parser.parse_args()

 
    main(args)

    # if args.predict:
    #     print(f"Heads above 90%: {Calibrate.good_predictions['greater_than_90']}")
    #     print(f"Heads above 95%: {Calibrate.good_predictions['greater_than_95']}")
    #     print(f"Heads above 97%: {Calibrate.good_predictions['greater_than_97']}")
    #     print(f"Target head predicted: {Calibrate.target_head_predicted}")
    #     print(f"Total heads: {Calibrate.total_heads}")
    #     print(f"Percentage of target heads predicted: {Calibrate.target_head_predicted / Calibrate.total_heads}")