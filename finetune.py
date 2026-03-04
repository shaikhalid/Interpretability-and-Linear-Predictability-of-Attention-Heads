import time
import argparse
import torch
import numpy as np

from tqdm import tqdm
import random
from transformers import AutoTokenizer
from models.modelling_llama3_viz import (
    LlamaForCausalLM as Llama3ForCausalLM,
    LlamaAttention as Llama3Attention,
)
from models.modelling_mistral import MistralForCausalLM, MistralAttention
from models.modelling_gptj import GPTJForCausalLM, GPTJAttention
from models.modelling_gemma2 import Gemma2ForCausalLM, Gemma2Attention
from models.llama3_modelling_aug_collect import *
from models.mistral_modelling_aug import *
from models.gptj_modelling_aug import *
from models.gemma2_modelling_aug import *
from models.calibrate import Calibrate
from utils.data_utils import *
from peft import get_peft_model, LoraConfig
from transformers import Trainer, TrainingArguments
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import json
import os
from transformers import pipeline
import torch.distributed as dist
from lm_eval import evaluator, models
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM
os.environ["WANDB_DISABLED"] = "true"

choices = ["A", "B", "C", "D", "E"]

models_to_layers = {
    "llama2": 32,
    "llama3": 32,
    "llama": 32,
    "mistral": 32,
    "gemma2-2b": 26,
    "llama3-3b": 32,
    "gemma2": 26
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load(ckpt_dir, model_type):
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=False,
        cache_dir="../.cache/",
        padding_side="left",
    )
    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.bos_token_id = 1
    Calibrate.tokenizer = tokenizer


    if "llama" in model_type:
        model = Llama3ForCausalLM.from_pretrained(
            ckpt_dir,
            low_cpu_mem_usage=True,
            cache_dir="../.cache",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model.to("cuda")

    elif "mistral" in model_type:
        model = MistralForCausalLM.from_pretrained(
            ckpt_dir,
            low_cpu_mem_usage=True,
            cache_dir="../.cache",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model.to("cuda")

    elif "gptj" in model_type:
        model = GPTJForCausalLM.from_pretrained(
            ckpt_dir,
            low_cpu_mem_usage=True,
            cache_dir="../.cache",
            attn_implementation="eager",
        )
        model.to("cuda")
    
    elif "gemma2-2b" in model_type:
        model = Gemma2ForCausalLM.from_pretrained(
            ckpt_dir,
            low_cpu_mem_usage=True,
            cache_dir="../.cache",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model.to("cuda")

    print(model)

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def prepare_input(tokenizer, prompts, padding="max_length"):
    input_tokens = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        max_length=args.context_length,
        truncation=True,
        padding=padding,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda")

    return input_tokens

def compute_metrics(args, results: dict, total_num: int) -> float:
    total_acc = 0
    accs = []
    for name, correct in results.items():
        acc = correct / total_num
        total_acc += correct
        print("ACC-%s: %.10f" % (name, acc))
    print("ACC-all: %.10f" % (total_acc / total_num))

    return total_acc / total_num
            
# taken from Aplaca LoRA: Stanford NLP
def generate_prompt(
    sentence,
    label,
) -> str:
    # returns the full prompt from instruction and optional input
    # if a label (=response, =output) is provided, it's also appended.
    if label is not None:
        if isinstance(label, int):
            res = f"{sentence}: {choices[label]}"
        else:
            res = f"{sentence}: {label}"
    else:
        res = sentence
    return res

def tokenize(prompt, tokenizer, padding="max_length"):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.context_length,
        padding=padding,
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_training_prompt(data_point, tokenizer, train_on_inputs=True):
    full_prompt = generate_prompt(
        data_point["sentence"],
        data_point["label"],
    )
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    # only compatible for single token generation

    if not train_on_inputs:
        if isinstance(data_point["label"], int):
            tokenized_label = tokenize(choices[data_point["label"]], tokenizer, padding=False)
        else:
            tokenized_label = tokenize(data_point["label"].strip(), tokenizer, padding=False)
        length = len(tokenized_label["input_ids"])-1
        tokenized_full_prompt["labels"] = [-100]*(len(tokenized_full_prompt["input_ids"]) - length) + tokenized_full_prompt["input_ids"][-length:]
   
    return tokenized_full_prompt

def fine_tune_model_with_lora(
    model,
    dataset,
    tokenizer,
    target_layers,
    target_modules,
    batch_size=8,
    num_epochs=3,
    lr=2e-4,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05
):
    # Configure LoRA

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        layers_to_transform=target_layers,
        bias="none",
        inference_mode=False,
    )
    
    model = get_peft_model(model, peft_config)

    # Prepare training dataset
    train_dataset = []
    for data_point in dataset:
        tokenized_data_point = generate_and_tokenize_training_prompt(
            data_point, tokenizer, train_on_inputs=False
        )
        train_dataset.append(tokenized_data_point)

    # Define training arguments
    if target_layers is None:
        run_name = f"finetune_baseline_{args.model_type}_{args.dataset}_log_term10"
    else:
        run_name = f"finetune_csaf_{args.model_type}_{args.dataset}_log_term10"

    training_args = TrainingArguments(
        output_dir="./lora_finetuned",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        optim="adamw_torch",  # Explicitly set optimizer to AdamW
        report_to=None,  # Enable logging to Weights & Biases
        # deepspeed="ds_config.json"
    )

    # Define compute metrics function
    def compute_metrics_for_trainer(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # Strip predictions and labels before comparison
        predictions = np.array([str(p).strip() for p in predictions])
        labels = np.array([str(l).strip() for l in labels])
        return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics_for_trainer,
    )

    # Train the model
    trainer.train()

    return model


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

    
    # Default tasks if none specified
    tasks = ['gsm8k_cot_llama']
    print(f"Evaluating model on tasks: {tasks}")
    
    # Create a HF model adapter for lm-evaluation-harness
    model_args = {
        "device": "cuda",
        "batch_size": batch_size,
        "pretrained": model
    }
    
    # Prepare the model for evaluation
    model.eval()
    
    # Define the HuggingFace model adapter
    eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda:0",
        batch_size=batch_size,
        max_length=args.context_length,
        dtype="float16",
        use_cache=True
    )
    # Run the evaluation
    print("Starting evaluation with lm-evaluation-harness...")
    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        num_fewshot=0,
    )

    if args.save_kv_cache:
        print("Saving KV cache")
        Llama3Attention.forward = llama3_atten_aug_forward_collect
        Calibrate.set_pickle_kv_cache(True)
        Calibrate.matrix_to_pickle = args.matrix_to_pickle
        # collecting kv cache for relationship analysis
        eval_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            device="cuda:0",
            batch_size=1,
            max_length=args.context_length,
            dtype="float16",
            use_cache=True
        )
        evaluator.simple_evaluate(
            model=eval_model,
            tasks=tasks,
            num_fewshot=0,
            limit=args.limit_samples,
        )
        
        
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
    print("SAVE MODEL ", args.save_adapter)
    set_random_seed(42)
    Calibrate.dataset = args.dataset.replace("/", "_")
    Calibrate.model_alias = args.alias
    Calibrate.model = args.ckpt_dir
    model, tokenizer = load(args.ckpt_dir, args.model_type)

   
    dataset = get_math_dataset(args.dataset)
    training_dataset = get_formatted_training_math_dataset(
        dataset, args.few_shot_number, args.finetune_sample_size
    )

    target_layers = (
        args.target_layers.split(",") if args.target_layers else None
    )
    target_layers = (
        [int(layer) for layer in target_layers] if target_layers else None
    )

    model = fine_tune_model_with_lora(
        model,
        training_dataset,
        tokenizer,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_layers=target_layers,
        target_modules=['q_proj', 'v_proj', 'k_proj'],
    )
    

    if args.save_adapter:
        print("saving the adapter")
        model.save_pretrained("./saved_adapters/finetuned_model")


    start_time = time.time()
    batch_size = 8
    model.eval()
    with torch.no_grad():
        acc = evaluate_with_lm_harness(model, tokenizer, tasks=[args.dataset], batch_size=batch_size)

    end_time = time.time()

    # log the results, task type, dataset, model type, and accuracy
    with open(args.output_dir, "a") as f:
        f.write(
            f"task_type={args.task_type} dataset={args.dataset} model={args.model_type} acc={acc} top_k={args.topk} learning_rate={args.lr} target_layers={target_layers} ft_sample_size={args.finetune_sample_size} context_length={args.context_length} random_layers={args.random_layers}\n"
        )
        
    print("Total run time: %.2f" % (end_time - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--few_shot_number", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ARCC")
    parser.add_argument(
        "--output_dir", type=str, default="./calibration_results/cal.log"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_layers", type=str, default=None)
    parser.add_argument("--finetune_sample_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--save_adapter", type=int, default=0)
    parser.add_argument("--save_kv_cache", type=bool, default=False)
    parser.add_argument("--matrix_to_pickle", type=str, default="k", choices=["q", "k", "v"],
                       help="Which matrix to pickle when kv_collection is enabled (q, k, or v)")
    parser.add_argument("--task_type", type=str, default="inference")
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--apply_chat_template", type=bool, default=False)
    parser.add_argument("--limit_samples", type=int, default=100)

    args = parser.parse_args()
    print(args)
        
    main(args)
    