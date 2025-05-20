import json
import pandas as pd
import argparse
import os
import ast
import re
import random
import torch
import numpy as np
from tqdm import tqdm
import bitsandbytes as bnb
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, set_seed, BitsAndBytesConfig
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = DeprecationWarning
from cappr.huggingface.classify import predict_proba
logging.set_verbosity_error()

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

MAX_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.95
DO_SAMPLE = True

BATCH_SIZE = 50

AVAILABLE_MODELS = {
    "llama3_2_3b_it": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.2-3b-instruct"
    },
    "llama3_1_8b_it": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.1-8b-instruct"
    },
    "aya_expanse_8b": {
        "model": "CohereForAI/aya-expanse-8b",
        "cache_dir": "/scratch/craj/model_cache/aya-expanse-8b"
    },
    "gemma_3_27b_it": {
        "model": "google/gemma-3-27b-it",
        "cache_dir": "/scratch/craj/model_cache/gemma-3-27b-instruct"
    },
    "qwen_32b": {
        "model": "Qwen/QwQ-32B",
        "cache_dir": "/scratch/craj/model_cache/qwen-32b"
    },
    "llama3_3_70b_it": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.3-70b-instruct"
    }
}

parser = argparse.ArgumentParser(description="Generate outputs with LLaMA 3 8B Instruct using cappr.")
parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
parser.add_argument("--mode", type=str, choices=["success", "failure"], default="success", help="Mode to run: success or failure")
parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), default="llama3_2_3b_it", help="Select the model to use")
parser.add_argument("--domains", nargs="+", default=None, help="Domains to process (e.g., education workplace or 'all')")
parser.add_argument("--dimension", type=str, choices=["nationality", "race", "religion"], default="nationality", help="Select dimension")
parser.add_argument("--limit", type=int, default=None, help="Limit number of new entries to process")

args = parser.parse_args()

MODE = args.mode
dimension_str = args.dimension
selected_model = AVAILABLE_MODELS[args.model]
MODEL, MODEL_CACHE_DIR = selected_model["model"], selected_model["cache_dir"]

# Use pre-generated expanded JSON files
EXPANDED_JSON_FILE = f"../../data/closed_ended/single_actor/{MODE}_{dimension_str}_all_5_names_expanded.json"

output_dir = f"../../outputs/closed_ended/single_actor/{dimension_str}/{args.model}"
os.makedirs(output_dir, exist_ok=True)
PARTIAL_CSV_FILE = f"{output_dir}/closed_ended_{MODE}_{args.model}_{dimension_str}_all_{args.runs}_runs_partial.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODEL_CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map="auto", cache_dir=MODEL_CACHE_DIR)

if MODE == "success":
    option_keys = ["opt1_higheffort", "opt2_highability", "opt3_easytask", "opt4_goodluck"]
else:
    option_keys = ["opt1_loweffort", "opt2_lowability", "opt3_difficulttask", "opt4_badluck"]

# Load expanded JSON
with open(EXPANDED_JSON_FILE, "r") as f:
    all_entries = json.load(f)
print(f"✅ Loaded {len(all_entries)} entries from {EXPANDED_JSON_FILE}")

# ✅ Filter domains if specified (other than "all")
if args.domains and args.domains != ["all"]:
    all_entries = [e for e in all_entries if e["domain"] in args.domains]
    print(f"✅ Filtered to domains {args.domains}, {len(all_entries)} entries left")


def safe_literal_eval(x):
    if isinstance(x, dict): return x
    if isinstance(x, str) and x.strip().startswith("{"):
        cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', x)
        try: return ast.literal_eval(cleaned)
        except: return {}
    return x

# Resume filtering
if os.path.exists(PARTIAL_CSV_FILE):
    print(f"🔁 Resuming from: {PARTIAL_CSV_FILE}")
    prev_df = pd.read_csv(PARTIAL_CSV_FILE)
    if "option_probs" in prev_df.columns:
        prev_df["option_probs"] = prev_df["option_probs"].apply(safe_literal_eval)
    if "shuffled_options" in prev_df.columns:
        prev_df["shuffled_options"] = prev_df["shuffled_options"].apply(safe_literal_eval)
    processed_ids = set(prev_df["set_id"] + "_" + prev_df["name"])
    all_entries = [e for e in all_entries if e["set_id"] + "_" + e["name"] not in processed_ids]
    all_runs_results = [prev_df]
else:
    all_runs_results = []

if args.limit:
    all_entries = all_entries[:args.limit]
print(f"⚙️ Processing {len(all_entries)} new entries")

def get_completion_with_cappr(model, tokenizer, entry_data):
    try:
        options = entry_data["shuffled_options"]
        prompt = (f"Respond by choosing one of the provided options as the reason to the statement.\n\n"
                  f"### Statement:\n{entry_data['initial_prompt']}\n\n"
                  f"Options:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n### Response:\n")
        pred_probs = predict_proba(prompt, completions=options, model_and_tokenizer=(model, tokenizer), end_of_prompt="", batch_size=1)
        pred_probs_rounded = pred_probs.round(2)
        option_probs = {chr(65 + i): float(prob) for i, prob in enumerate(pred_probs_rounded)}
        chosen_idx = pred_probs.argmax()
        chosen_letter = chr(65 + chosen_idx)
        return {
            "initial_prompt": entry_data["initial_prompt"],
            "chosen_letter": chosen_letter,
            "chosen_key": entry_data["shuffled_option_mapping"][chosen_letter],
            "chosen_sentence": options[chosen_idx],
            "shuffled_options": entry_data["shuffled_option_mapping"],
            "option_probs": option_probs,
        }
    except Exception as e:
        print(f"❌ Error: {e}\n⚠️ Failed entry: {entry_data}")
        return None

def expand_option_probs(row):
    if pd.isna(row.get("option_probs")) or not isinstance(row["option_probs"], dict): return {}
    return {row["shuffled_options"][k]: v for k, v in row["option_probs"].items() if k in row["shuffled_options"]}

def run_experiment(data, model, tokenizer, run_number, seed):
    results = []
    for idx, entry in tqdm(enumerate(data), total=len(data), desc="Processing"):
        original_options = OrderedDict([(key, entry[key]) for key in option_keys])
        random.seed(seed + idx)
        shuffled = list(original_options.items()); random.shuffle(shuffled)
        shuffled_options = [v for k, v in shuffled]
        shuffled_mapping = {chr(65 + i): k for i, (k, _) in enumerate(shuffled)}
        entry_data = {
            "index": idx, "initial_prompt": entry["initial_prompt"], "original_options": dict(original_options),
            "shuffled_options": shuffled_options, "shuffled_option_mapping": shuffled_mapping,
            "gender": entry["gender"], args.dimension: entry["dimension"], "name": entry["name"], "set_id": entry["set_id"]
        }
        result = get_completion_with_cappr(model, tokenizer, entry_data)
        if result:
            result.update({"run": run_number, "seed": seed, "gender": entry["gender"], args.dimension: entry["dimension"],
                           "name": entry["name"], "set_id": entry["set_id"], "domain": entry["domain"]})
            results.append(result)
    df = pd.DataFrame(results)
    if not df.empty:
        expanded = pd.DataFrame(df.apply(expand_option_probs, axis=1).tolist())
        df = pd.concat([df, expanded], axis=1)
        df.to_csv(PARTIAL_CSV_FILE, mode="a", index=False, header=not os.path.exists(PARTIAL_CSV_FILE))
    return df

for run in tqdm(range(args.runs), total=args.runs, desc="Runs"):
    seed = random.randint(0, 100000)
    set_seed(seed)
    run_experiment(all_entries, model, tokenizer, run + 1, seed)

# Final merge
if os.path.exists(PARTIAL_CSV_FILE):
    final_df = pd.read_csv(PARTIAL_CSV_FILE)
    if "option_probs" in final_df.columns:
        final_df["option_probs"] = final_df["option_probs"].apply(safe_literal_eval)
    if "shuffled_options" in final_df.columns:
        final_df["shuffled_options"] = final_df["shuffled_options"].apply(safe_literal_eval)
else:
    final_df = pd.concat(all_runs_results)

final_csv = f"{output_dir}/closed_ended_{MODE}_{args.model}_{dimension_str}_all_{args.runs}_runs.csv"
final_df.to_csv(final_csv, index=False)
print(f"✅ Final results saved to {final_csv}")