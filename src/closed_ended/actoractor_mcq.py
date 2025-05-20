# import json
# import pandas as pd
# import argparse
# import os
# import ast
# import re
# import random
# import torch
# import numpy as np
# from tqdm import tqdm
# import bitsandbytes as bnb
# from collections import OrderedDict
# from transformers import AutoTokenizer, AutoModelForCausalLM, logging, set_seed, BitsAndBytesConfig
# if not hasattr(np, "VisibleDeprecationWarning"):
#     np.VisibleDeprecationWarning = DeprecationWarning
# from cappr.huggingface.classify import predict_proba

# logging.set_verbosity_error()

# GLOBAL_SEED = 42
# random.seed(GLOBAL_SEED)
# torch.manual_seed(GLOBAL_SEED)

# MAX_TOKENS = 200
# TEMPERATURE = 0.7
# TOP_P = 0.95
# DO_SAMPLE = True

# BATCH_SIZE = 50

# AVAILABLE_MODELS = {
#     "llama3_2_3b_it": {"model": "meta-llama/Llama-3.2-3B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.2-3b-instruct"},
#     "llama3_1_8b_it": {"model": "meta-llama/Llama-3.1-8B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.1-8b-instruct"},
#     "aya_expanse_8b": {"model": "CohereForAI/aya-expanse-8b", "cache_dir": "/scratch/craj/model_cache/aya-expanse-8b"},
#     "gemma_3_27b_it": {"model": "google/gemma-3-27b-it", "cache_dir": "/scratch/craj/model_cache/gemma-3-27b-instruct"},
#     "qwen_32b": {"model": "Qwen/QwQ-32B", "cache_dir": "/scratch/craj/model_cache/qwen-32b"},
#     "llama3_3_70b_it": {"model": "meta-llama/Llama-3.3-70B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.3-70b-instruct"}
# }

# parser = argparse.ArgumentParser(description="Generate outputs with actor-actor setting using cappr.")
# parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
# parser.add_argument("--mode", type=str, choices=["both_success", "both_failure", "success_failure"], default="both_success", help="Mode to run")
# parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), default="llama3_2_3b_it", help="Select the model to use")
# parser.add_argument("--domains", nargs="+", default=None, help="Domains to process or 'all'")
# parser.add_argument("--dimension", type=str, choices=["nationality", "race", "religion"], default="nationality", help="Select dimension")
# parser.add_argument("--limit", type=int, default=None, help="Limit number of new entries to process")

# args = parser.parse_args()

# MODE = args.mode
# dimension_str = args.dimension
# selected_model = AVAILABLE_MODELS[args.model]
# MODEL, MODEL_CACHE_DIR = selected_model["model"], selected_model["cache_dir"]

# EXPANDED_JSON_FILE = f"../../data/closed_ended/actor_actor/{MODE}_{dimension_str}_all_limited.json"
# output_dir = f"../../outputs/closed_ended/actor_actor/{dimension_str}/{args.model}"
# os.makedirs(output_dir, exist_ok=True)
# PARTIAL_CSV_FILE = f"{output_dir}/closed_ended_{MODE}_{args.model}_{dimension_str}_all_{args.runs}_runs_partial.csv"

# tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODEL_CACHE_DIR)
# tokenizer.pad_token = tokenizer.eos_token
# quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map="auto", cache_dir=MODEL_CACHE_DIR)

# if MODE == "both_success":
#     option_keys_x = ["optX1_higheffort", "optX2_highability", "optX3_easytask", "optX4_goodluck"]
#     option_keys_y = ["optY1_higheffort", "optY2_highability", "optY3_easytask", "optY4_goodluck"]
# elif MODE == "both_failure":
#     option_keys_x = ["optX1_loweffort", "optX2_lowability", "optX3_difficulttask", "optX4_badluck"]
#     option_keys_y = ["optY1_loweffort", "optY2_lowability", "optY3_difficulttask", "optY4_badluck"]
# else:
#     option_keys_x = ["optX1_higheffort", "optX2_highability", "optX3_easytask", "optX4_goodluck"]
#     option_keys_y = ["optY1_loweffort", "optY2_lowability", "optY3_difficulttask", "optY4_badluck"]

# with open(EXPANDED_JSON_FILE, "r") as f:
#     all_entries = json.load(f)
# print(f"✅ Loaded {len(all_entries)} entries from {EXPANDED_JSON_FILE}")

# if args.domains and args.domains != ["all"]:
#     all_entries = [e for e in all_entries if e["domain"] in args.domains]
#     print(f"✅ Filtered to domains {args.domains}, {len(all_entries)} entries left")

# if os.path.exists(PARTIAL_CSV_FILE):
#     print(f"🔁 Resuming from: {PARTIAL_CSV_FILE}")
#     prev_df = pd.read_csv(PARTIAL_CSV_FILE)
#     processed_ids = set(prev_df["set_id"])
#     all_entries = [e for e in all_entries if e["set_id"] not in processed_ids]
#     all_runs_results = [prev_df]
# else:
#     all_runs_results = []

# if args.limit:
#     all_entries = all_entries[:args.limit]
# print(f"⚙️ Processing {len(all_entries)} new entries")

# def get_completion_with_cappr(model, tokenizer, entry_data):
#     try:
#         x_options = entry_data["shuffled_X_options"]
#         y_options = entry_data["shuffled_Y_options"]
#         prompt = (f"Respond by choosing one of the provided options for each person.\n\n"
#                   f"### Statement:\n{entry_data['initial_prompt']}\n\n"
#                   f"Options for {entry_data['name1']}:\n"
#                   f"A. {x_options[0]}\nB. {x_options[1]}\nC. {x_options[2]}\nD. {x_options[3]}\n\n"
#                   f"Options for {entry_data['name2']}:\n"
#                   f"E. {y_options[0]}\nF. {y_options[1]}\nG. {y_options[2]}\nH. {y_options[3]}\n\n"
#                   f"### Response:\nX: <A-D>\nY: <E-H>")
#         pred_probs = predict_proba(prompt, completions=x_options + y_options, model_and_tokenizer=(model, tokenizer), end_of_prompt="", batch_size=1)
#         # Separate X and Y probabilities
#         x_probs_raw = pred_probs[:4]
#         y_probs_raw = pred_probs[4:]
#         # Normalize X and Y separately
#         x_probs = x_probs_raw / x_probs_raw.sum()
#         y_probs = y_probs_raw / y_probs_raw.sum()
#         # Round for reporting
#         x_probs_rounded = x_probs.round(2)
#         y_probs_rounded = y_probs.round(2)

#         option_probs = {}
#         for i in range(4):
#             option_probs[entry_data["shuffled_X_mapping"][chr(65 + i)]] = float(x_probs_rounded[i])
#         for i in range(4):
#             option_probs[entry_data["shuffled_Y_mapping"][chr(69 + i)]] = float(y_probs_rounded[i])

#         x_idx = x_probs.argmax()
#         y_idx = y_probs.argmax()
#         x_letter = chr(65 + x_idx)
#         y_letter = chr(69 + y_idx)
#         return {
#             "initial_prompt": entry_data["initial_prompt"],
#             "X_choice_letter": x_letter,
#             "Y_choice_letter": y_letter,
#             "X_choice_key": entry_data["shuffled_X_mapping"][x_letter],
#             "Y_choice_key": entry_data["shuffled_Y_mapping"][y_letter],
#             "X_choice_text": x_options[x_idx],
#             "Y_choice_text": y_options[y_idx - 4],
#             "option_probs": option_probs
#         }
#     except Exception as e:
#         print(f"❌ Error: {e}\n⚠️ Failed entry: {entry_data}")
#         return None

# def run_experiment(data, model, tokenizer, run_number, seed):
#     results = []
#     for idx, entry in tqdm(enumerate(data), total=len(data), desc="Processing"):
#         x_options = OrderedDict([(k, entry[k]) for k in option_keys_x])
#         y_options = OrderedDict([(k, entry[k]) for k in option_keys_y])
#         random.seed(seed + idx)
#         x_items = list(x_options.items()); random.shuffle(x_items)
#         y_items = list(y_options.items()); random.shuffle(y_items)
#         x_shuffled = [v for k, v in x_items]
#         y_shuffled = [v for k, v in y_items]
#         x_mapping = {chr(65 + i): k for i, (k, _) in enumerate(x_items)}
#         y_mapping = {chr(69 + i): k for i, (k, _) in enumerate(y_items)}
#         entry_data = {
#             "index": idx, "initial_prompt": entry["initial_prompt"],
#             "shuffled_X_options": x_shuffled, "shuffled_X_mapping": x_mapping,
#             "shuffled_Y_options": y_shuffled, "shuffled_Y_mapping": y_mapping,
#             "gender_pair": entry["gender_pair"], "domain": entry["domain"], "length": entry["length"],
#             "name1": entry["name1"], "name2": entry["name2"], "dimension1": entry["dimension1"],
#             "dimension2": entry["dimension2"], "set_id": entry["set_id"]
#         }
#         result = get_completion_with_cappr(model, tokenizer, entry_data)
#         if result:
#             result.update({
#                 "run": run_number, "seed": seed, "gender_pair": entry["gender_pair"], "domain": entry["domain"],
#                 "length": entry["length"], "name1": entry["name1"], "name2": entry["name2"],
#                 "dimension1": entry["dimension1"], "dimension2": entry["dimension2"], "set_id": entry["set_id"],
#                 # ➜ use clean string, not json.dumps()
#                 "shuffled_X_options": ", ".join([f"{k}:{v}" for k, v in x_mapping.items()]),
#                 "shuffled_Y_options": ", ".join([f"{k}:{v}" for k, v in y_mapping.items()]),
#                 # ➜ store full option_probs as JSON string
#                 "option_probs": result["option_probs"]
#             })
#             results.append(result)

#     df = pd.DataFrame(results)
#     # Step 1: Extract option_probs_df
#     option_probs_df = pd.DataFrame(df["option_probs"].apply(
#         lambda x: x if isinstance(x, dict) else json.loads(x)
#     ).tolist())

#     if not option_probs_df.empty:
#         # Step 2: Merge first → so columns exist in df
#         df = pd.concat([df, option_probs_df], axis=1)

#         # Step 3: Create flattened string (now columns exist)
#         df["option_probs"] = df[option_probs_df.columns].apply(
#             lambda row: ", ".join([f"{k}={row[k]:.2f}" for k in option_probs_df.columns]), axis=1
#         )

#         # Step 4: Define final column order
#         final_columns = [
#             "initial_prompt", "X_choice_letter", "Y_choice_letter", "X_choice_key", "Y_choice_key",
#             "X_choice_text", "Y_choice_text", "run", "seed", "gender_pair", "domain", "length",
#             "name1", "name2", "dimension1", "dimension2", "set_id",
#             "shuffled_X_options", "shuffled_Y_options", "option_probs"
#         ] + sorted(option_probs_df.columns.tolist())

#         df = df[final_columns]

#     # ➜ save partial CSV
#     df.to_csv(PARTIAL_CSV_FILE, mode="a", index=False, header=not os.path.exists(PARTIAL_CSV_FILE))
#     return df


# for run in tqdm(range(args.runs), total=args.runs, desc="Runs"):
#     seed = random.randint(0, 100000)
#     set_seed(seed)
#     run_experiment(all_entries, model, tokenizer, run + 1, seed)

# if os.path.exists(PARTIAL_CSV_FILE):
#     final_df = pd.read_csv(PARTIAL_CSV_FILE)
# elif all_runs_results:
#     final_df = pd.concat(all_runs_results)
# else:
#     print("⚠️ No runs completed successfully. Exiting.")
#     exit(1)

# final_csv = f"{output_dir}/closed_ended_{MODE}_{args.model}_{dimension_str}_all_{args.runs}_runs.csv"
# final_df.to_csv(final_csv, index=False)
# print(f"✅ Final results saved to {final_csv}")




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

BATCH_SIZE = 10000

AVAILABLE_MODELS = {
    "llama3_2_3b_it": {"model": "meta-llama/Llama-3.2-3B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.2-3b-instruct"},
    "llama3_1_8b_it": {"model": "meta-llama/Llama-3.1-8B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.1-8b-instruct"},
    "aya_expanse_8b": {"model": "CohereForAI/aya-expanse-8b", "cache_dir": "/scratch/craj/model_cache/aya-expanse-8b"},
    "gemma_3_27b_it": {"model": "google/gemma-3-27b-it", "cache_dir": "/scratch/craj/model_cache/gemma-3-27b-instruct"},
    "qwen_32b": {"model": "Qwen/QwQ-32B", "cache_dir": "/scratch/craj/model_cache/qwen-32b"},
    "llama3_3_70b_it": {"model": "meta-llama/Llama-3.3-70B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.3-70b-instruct"}
}

parser = argparse.ArgumentParser(description="Generate outputs with actor-actor setting using cappr.")
parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
parser.add_argument("--mode", type=str, choices=["both_success", "both_failure", "success_failure"], default="both_success", help="Mode to run")
parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), default="llama3_2_3b_it", help="Select the model to use")
parser.add_argument("--domains", nargs="+", default=None, help="Domains to process or 'all'")
parser.add_argument("--dimension", type=str, choices=["nationality", "race", "religion"], default="nationality", help="Select dimension")
parser.add_argument("--limit", type=int, default=None, help="Limit number of new entries to process")

args = parser.parse_args()

MODE = args.mode
dimension_str = args.dimension
selected_model = AVAILABLE_MODELS[args.model]
MODEL, MODEL_CACHE_DIR = selected_model["model"], selected_model["cache_dir"]

EXPANDED_JSON_FILE = f"../../data/closed_ended/actor_actor/{MODE}_{dimension_str}_all_limited.json"
output_dir = f"../../outputs/closed_ended/actor_actor/{dimension_str}/{args.model}"
# output_dir = f"../../outputs/closed_ended/actor_actor/{dimension_str}/test"
os.makedirs(output_dir, exist_ok=True)
PARTIAL_CSV_FILE = f"{output_dir}/closed_ended_{MODE}_{args.model}_{dimension_str}_all_{args.runs}_runs_partial.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODEL_CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map="auto", cache_dir=MODEL_CACHE_DIR)

if MODE == "both_success":
    option_keys_x = ["optX1_higheffort", "optX2_highability", "optX3_easytask", "optX4_goodluck"]
    option_keys_y = ["optY1_higheffort", "optY2_highability", "optY3_easytask", "optY4_goodluck"]
elif MODE == "both_failure":
    option_keys_x = ["optX1_loweffort", "optX2_lowability", "optX3_difficulttask", "optX4_badluck"]
    option_keys_y = ["optY1_loweffort", "optY2_lowability", "optY3_difficulttask", "optY4_badluck"]
else:
    option_keys_x = ["optX1_higheffort", "optX2_highability", "optX3_easytask", "optX4_goodluck"]
    option_keys_y = ["optY1_loweffort", "optY2_lowability", "optY3_difficulttask", "optY4_badluck"]

with open(EXPANDED_JSON_FILE, "r") as f:
    all_entries = json.load(f)
print(f"✅ Loaded {len(all_entries)} entries from {EXPANDED_JSON_FILE}")

if args.domains and args.domains != ["all"]:
    all_entries = [e for e in all_entries if e["domain"] in args.domains]
    print(f"✅ Filtered to domains {args.domains}, {len(all_entries)} entries left")

if os.path.exists(PARTIAL_CSV_FILE):
    print(f"🔁 Resuming from: {PARTIAL_CSV_FILE}")
    prev_df = pd.read_csv(PARTIAL_CSV_FILE)
    processed_ids = set(prev_df["set_id"])
    all_entries = [e for e in all_entries if e["set_id"] not in processed_ids]
    all_runs_results = [prev_df]
else:
    all_runs_results = []

if args.limit:
    all_entries = all_entries[:args.limit]
print(f"⚙️ Processing {len(all_entries)} new entries")

def get_completion_with_cappr(model, tokenizer, entry_data):
    try:
        x_options = entry_data["shuffled_X_options"]
        y_options = entry_data["shuffled_Y_options"]
        prompt = (f"Respond by choosing one of the provided options for each person.\n\n"
                  f"### Statement:\n{entry_data['initial_prompt']}\n\n"
                  f"Options for {entry_data['name1']}:\n"
                  f"A. {x_options[0]}\nB. {x_options[1]}\nC. {x_options[2]}\nD. {x_options[3]}\n\n"
                  f"Options for {entry_data['name2']}:\n"
                  f"E. {y_options[0]}\nF. {y_options[1]}\nG. {y_options[2]}\nH. {y_options[3]}\n\n"
                  f"### Response:\nX: <A-D>\nY: <E-H>")
        pred_probs = predict_proba(prompt, completions=x_options + y_options, model_and_tokenizer=(model, tokenizer), end_of_prompt="", batch_size=1)
        # Separate X and Y probabilities
        x_probs_raw = pred_probs[:4]
        y_probs_raw = pred_probs[4:]
        # Normalize X and Y separately
        x_probs = x_probs_raw / x_probs_raw.sum()
        y_probs = y_probs_raw / y_probs_raw.sum()
        # Round for reporting
        x_probs_rounded = x_probs.round(2)
        y_probs_rounded = y_probs.round(2)

        option_probs = {}
        for i in range(4):
            option_probs[entry_data["shuffled_X_mapping"][chr(65 + i)]] = float(x_probs_rounded[i])
        for i in range(4):
            option_probs[entry_data["shuffled_Y_mapping"][chr(69 + i)]] = float(y_probs_rounded[i])

        x_idx = x_probs.argmax()
        y_idx = y_probs.argmax()
        x_letter = chr(65 + x_idx)
        y_letter = chr(69 + y_idx)
        return {
            "initial_prompt": entry_data["initial_prompt"],
            "X_choice_letter": x_letter,
            "Y_choice_letter": y_letter,
            "X_choice_key": entry_data["shuffled_X_mapping"][x_letter],
            "Y_choice_key": entry_data["shuffled_Y_mapping"][y_letter],
            "X_choice_text": x_options[x_idx],
            "Y_choice_text": y_options[y_idx - 4],
            "option_probs": option_probs
        }
    except Exception as e:
        print(f"❌ Error: {e}\n⚠️ Failed entry: {entry_data}")
        return None

def save_partial_results(results):
    df = pd.DataFrame(results)

    option_probs_df = pd.DataFrame(df["option_probs"].apply(
        lambda x: x if isinstance(x, dict) else json.loads(x)
    ).tolist())

    if not option_probs_df.empty:
        df = pd.concat([df, option_probs_df], axis=1)
        df["option_probs"] = df[option_probs_df.columns].apply(
            lambda row: ", ".join([f"{k}={row[k]:.2f}" for k in option_probs_df.columns]), axis=1
        )

        final_columns = [
            "initial_prompt", "X_choice_letter", "Y_choice_letter", "X_choice_key", "Y_choice_key",
            "X_choice_text", "Y_choice_text", "run", "seed", "gender_pair", "domain", "length",
            "name1", "name2", "dimension1", "dimension2", "set_id",
            "shuffled_X_options", "shuffled_Y_options", "option_probs"
        ] + sorted(option_probs_df.columns.tolist())

        df = df[final_columns]

    df.to_csv(PARTIAL_CSV_FILE, mode="a", index=False, header=not os.path.exists(PARTIAL_CSV_FILE))


def run_experiment(data, model, tokenizer, run_number, seed):
    results = []
    for idx, entry in tqdm(enumerate(data), total=len(data), desc="Processing"):
        x_options = OrderedDict([(k, entry[k]) for k in option_keys_x])
        y_options = OrderedDict([(k, entry[k]) for k in option_keys_y])
        random.seed(seed + idx)
        x_items = list(x_options.items()); random.shuffle(x_items)
        y_items = list(y_options.items()); random.shuffle(y_items)
        x_shuffled = [v for k, v in x_items]
        y_shuffled = [v for k, v in y_items]
        x_mapping = {chr(65 + i): k for i, (k, _) in enumerate(x_items)}
        y_mapping = {chr(69 + i): k for i, (k, _) in enumerate(y_items)}
        entry_data = {
            "index": idx, "initial_prompt": entry["initial_prompt"],
            "shuffled_X_options": x_shuffled, "shuffled_X_mapping": x_mapping,
            "shuffled_Y_options": y_shuffled, "shuffled_Y_mapping": y_mapping,
            "gender_pair": entry["gender_pair"], "domain": entry["domain"], "length": entry["length"],
            "name1": entry["name1"], "name2": entry["name2"], "dimension1": entry["dimension1"],
            "dimension2": entry["dimension2"], "set_id": entry["set_id"]
        }
        result = get_completion_with_cappr(model, tokenizer, entry_data)
        if result:
            result.update({
                "run": run_number, "seed": seed, "gender_pair": entry["gender_pair"], "domain": entry["domain"],
                "length": entry["length"], "name1": entry["name1"], "name2": entry["name2"],
                "dimension1": entry["dimension1"], "dimension2": entry["dimension2"], "set_id": entry["set_id"],
                "shuffled_X_options": ", ".join([f"{k}:{v}" for k, v in x_mapping.items()]),
                "shuffled_Y_options": ", ".join([f"{k}:{v}" for k, v in y_mapping.items()]),
                "option_probs": result["option_probs"]
            })
            results.append(result)

        # ⏳ Write to partial CSV every BATCH_SIZE
        if len(results) >= BATCH_SIZE:
            save_partial_results(results)
            results = []

    # 💾 Flush remaining
    if results:
        save_partial_results(results)


for run in tqdm(range(args.runs), total=args.runs, desc="Runs"):
    seed = random.randint(0, 100000)
    set_seed(seed)
    run_experiment(all_entries, model, tokenizer, run + 1, seed)

if os.path.exists(PARTIAL_CSV_FILE):
    final_df = pd.read_csv(PARTIAL_CSV_FILE)
elif all_runs_results:
    final_df = pd.concat(all_runs_results)
else:
    print("⚠️ No runs completed successfully. Exiting.")
    exit(1)

final_csv = f"{output_dir}/closed_ended_{MODE}_{args.model}_{dimension_str}_all_{args.runs}_runs.csv"
final_df.to_csv(final_csv, index=False)
print(f"✅ Final results saved to {final_csv}")