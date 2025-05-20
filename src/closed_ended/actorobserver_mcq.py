import json
import pandas as pd
import argparse
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import bitsandbytes as bnb
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, set_seed, BitsAndBytesConfig
from cappr.huggingface.classify import predict_proba

if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = DeprecationWarning

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
    "llama3_2_3b_it": {"model": "meta-llama/Llama-3.2-3B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.2-3b-instruct"},
    "llama3_1_8b_it": {"model": "meta-llama/Llama-3.1-8B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.1-8b-instruct"},
    "aya_expanse_8b": {"model": "CohereForAI/aya-expanse-8b", "cache_dir": "/scratch/craj/model_cache/aya-expanse-8b"},
    "gemma_3_27b_it": {"model": "google/gemma-3-27b-it", "cache_dir": "/scratch/craj/model_cache/gemma-3-27b-instruct"},
    "qwen_32b": {"model": "Qwen/QwQ-32B", "cache_dir": "/scratch/craj/model_cache/qwen-32b"},
    "llama3_3_70b_it": {"model": "meta-llama/Llama-3.3-70B-Instruct", "cache_dir": "/scratch/craj/model_cache/llama-3.3-70b-instruct"}
}

parser = argparse.ArgumentParser(description="Generate outputs with actor-observer setting using cappr.")
parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
parser.add_argument("--mode", type=str, choices=["success", "failure"], default="success", help="Mode to run")
parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), default="llama3_2_3b_it", help="Select the model to use")
parser.add_argument("--domains", nargs="+", default=None, help="Domains to process or 'all'")
parser.add_argument("--dimension", type=str, choices=["nationality", "race", "religion"], default="nationality", help="Select dimension")
parser.add_argument("--limit", type=int, default=None, help="Limit number of new entries to process")

args = parser.parse_args()

MODE = args.mode
dimension_str = args.dimension
selected_model = AVAILABLE_MODELS[args.model]
MODEL, MODEL_CACHE_DIR = selected_model["model"], selected_model["cache_dir"]

EXPANDED_JSON_FILE = f"../../data/closed_ended/actor_observer/{MODE}_{dimension_str}_all_limited.json"
output_dir = f"../../outputs/closed_ended/actor_observer/{dimension_str}/{args.model}"
os.makedirs(output_dir, exist_ok=True)
PARTIAL_CSV_FILE = f"{output_dir}/closed_ended_{MODE}_{args.model}_{dimension_str}_all_{args.runs}_runs_partial.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODEL_CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map="auto", cache_dir=MODEL_CACHE_DIR)

option_keys = ["opt1_higheffort", "opt2_highability", "opt3_easytask", "opt4_goodluck"] if MODE == "success" else ["opt1_loweffort", "opt2_lowability", "opt3_difficulttask", "opt4_badluck"]

with open(EXPANDED_JSON_FILE, "r") as f:
    all_entries = json.load(f)
print(f"✅ Loaded {len(all_entries)} entries from {EXPANDED_JSON_FILE}")

if args.domains and args.domains != ["all"]:
    all_entries = [e for e in all_entries if e["domain"] in args.domains]
    print(f"✅ Filtered to domains {args.domains}, {len(all_entries)} entries left")

for e in all_entries:
    e["set_id"] = f"{e['domain']}_{e['gender']}_{e['dimension1']}_{e['dimension2']}_{e['name1']}_{e['name2']}"


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
        options = entry_data["shuffled_options"]
        prompt = (f"Respond by choosing one of the provided options.\n\n"
                  f"### Statement:\n{entry_data['initial_prompt']}\n\n"
                  f"Options:\n"
                  f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n\n"
                  f"### Response:\n< A-D >")
        pred_probs = predict_proba(prompt, completions=options, model_and_tokenizer=(model, tokenizer), end_of_prompt="", batch_size=1)
        probs = pred_probs / pred_probs.sum()
        probs_rounded = probs.round(2)
        option_probs = {entry_data["shuffled_mapping"][chr(65 + i)]: float(probs_rounded[i]) for i in range(4)}
        idx = probs.argmax()
        letter = chr(65 + idx)
        return {
            "initial_prompt": entry_data["initial_prompt"],
            "choice_letter": letter,
            "choice_key": entry_data["shuffled_mapping"][letter],
            "choice_text": options[idx],
            "option_probs": option_probs
        }
    except Exception as e:
        print(f"❌ Error: {e}\n⚠️ Failed entry: {entry_data}")
        return None

def run_experiment(data, model, tokenizer, run_number, seed):
    results = []
    for idx, entry in tqdm(enumerate(data), total=len(data), desc="Processing"):
        options = OrderedDict([(k, entry[k]) for k in option_keys])
        random.seed(seed + idx)
        items = list(options.items()); random.shuffle(items)
        shuffled = [v for k, v in items]
        mapping = {chr(65 + i): k for i, (k, _) in enumerate(items)}
        entry_data = {
            "index": idx, 
            "initial_prompt": entry["initial_prompt"],
            "shuffled_options": shuffled, 
            "shuffled_mapping": mapping,
            "length": entry["length"],
            "gender": entry["gender"],  
            "domain": entry["domain"],  
            "name1": entry["name1"],  
            "name2": entry["name2"],  
            "dimension1": entry["dimension1"],  
            "dimension2": entry["dimension2"],  
            "y_opt_key": entry["y_opt_key"]
        }
        result = get_completion_with_cappr(model, tokenizer, entry_data)
        if result:
            result.update({
                "run": run_number, 
                "seed": seed, 
                "gender": entry["gender"], 
                "domain": entry["domain"],
                "length": entry["length"], 
                "name1": entry["name1"],  
                "name2": entry["name2"],  
                "dimension1": entry["dimension1"],  
                "dimension2": entry["dimension2"],  
                "y_opt_key": entry["y_opt_key"],
                "shuffled_options": ", ".join([f"{k}:{v}" for k, v in mapping.items()]),
                "option_probs": result["option_probs"]
            })
            results.append(result)

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
            "initial_prompt", "choice_letter", "choice_key", "choice_text", "run", "seed",
            "gender", "domain", "name1", "name2", "dimension1", "dimension2", "y_opt_key",
            "shuffled_options", "option_probs"
        ] + sorted(option_probs_df.columns.tolist())
        df = df[final_columns]

    df.to_csv(PARTIAL_CSV_FILE, mode="a", index=False, header=not os.path.exists(PARTIAL_CSV_FILE))
    return df

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






# from transformers import AutoTokenizer, AutoModelForCausalLM, logging, set_seed, BitsAndBytesConfig
# import pandas as pd
# import json
# from tqdm import tqdm
# import pandas as pd
# import argparse
# import os
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import torch
# import random
# from collections import OrderedDict
# from cappr.huggingface.classify import predict_proba
# import bitsandbytes as bnb
# import ast

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
#     "llama3_2_3b": {
#         "model": "meta-llama/Llama-3.2-3B-Instruct",
#         "cache_dir": "/scratch/craj/model_cache/llama-3.2-3b-instruct"
#     },
#     "llama3_1_8b": {
#         "model": "meta-llama/Llama-3.1-8B-Instruct",
#         "cache_dir": "/scratch/craj/model_cache/llama-3.1-8b-instruct"
#     },
#     "aya_expanse_8b": {
#         "model": "CohereForAI/aya-expanse-8b",
#         "cache_dir": "/scratch/craj/model_cache/aya-expanse-8b"
#     }
# }

# parser = argparse.ArgumentParser(description="Generate outputs with LLaMA 3 8B Instruct using cappr.")
# parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
# parser.add_argument("--mode", type=str, choices=["success", "failure"], default="success", help="Mode to run: success or failure")
# parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), default="llama3_2_3b", help="Select the model to use")
# parser.add_argument("--domains", nargs="+", default=None, help="Domains to process (e.g., education workplace or 'all')")
# parser.add_argument("--lengths", nargs="+", default=None, help="Lengths to process (e.g., short long or 'all')")

# args = parser.parse_args()

# MODE = args.mode
# SUCCESS_INPUT = f"../data/actor_observer/success.json"
# FAILURE_INPUT = f"../data/actor_observer/failure.json"
# NAMES_FILE = "../data/names/names.json"

# length_str = "all" if args.lengths is None else "_".join(args.lengths)
# domain_str = "all" if args.domains is None else "_".join(args.domains)
# OUTPUT_JSON_FILE = f"../data/actor_observer/{MODE}_{length_str}_{domain_str}_expanded.json"
# os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)

# selected_model = AVAILABLE_MODELS[args.model]
# MODEL = selected_model["model"]
# MODEL_CACHE_DIR = selected_model["cache_dir"]

# output_dir = f"../outputs/actor_observer/{args.model}"
# os.makedirs(output_dir, exist_ok=True)
# PARTIAL_CSV_FILE = f"{output_dir}/actor_observer_{MODE}_{args.model}_{length_str}_{args.runs}_runs_partial.csv"


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODEL_CACHE_DIR)
# tokenizer.pad_token = tokenizer.eos_token

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL,
#     quantization_config=quant_config,
#     device_map="auto",
#     cache_dir=MODEL_CACHE_DIR,
# )


# if MODE == "success":
#     INPUT_FILE = SUCCESS_INPUT
#     option_keys = ["opt1_higheffort", "opt2_highability", "opt3_easytask", "opt4_goodluck"]
# else:
#     INPUT_FILE = FAILURE_INPUT
#     option_keys = ["opt1_loweffort", "opt2_lowability", "opt3_difficulttask", "opt4_badluck"]


# with open(INPUT_FILE, "r") as f:
#     nested_input_data = json.load(f)

# # Extract all available domains and lengths
# all_domains = set()
# all_lengths = set()
# for gender_data in nested_input_data.values():
#     for domain, lengths in gender_data.items():
#         all_domains.add(domain)
#         all_lengths.update(lengths.keys())

# # Handle "all" case
# selected_domains = all_domains if args.domains == ["all"] else set(args.domains) if args.domains else None
# selected_lengths = all_lengths if args.lengths == ["all"] else set(args.lengths) if args.lengths else None


# with open(NAMES_FILE, "r") as f:
#     names_data = json.load(f)

# # def generate_name_pairs(names_data, gender):
# #     name_pairs = []

# #     # Get gender variations
# #     opposite_gender = "male" if gender == "female" else "female"

# #     for region1, group1 in names_data.items():
# #         group1_nested = group1.get(gender.capitalize(), [])
# #         group1_opposite = group1.get(opposite_gender.capitalize(), [])

# #         for name1 in group1_nested:
# #             for region2, group2 in names_data.items():
# #                 group2_same = group2.get(gender.capitalize(), [])
# #                 group2_opposite = group2.get(opposite_gender.capitalize(), [])

# #                 # same gender + same nationality
# #                 for name2 in group1_nested:
# #                     if name1 != name2:
# #                         name_pairs.append((name1, name2, gender, region1, gender, region1))

# #                 # different gender, same nationality
# #                 for name2 in group1_opposite:
# #                     if name1 != name2:
# #                         name_pairs.append((name1, name2, gender, region1, opposite_gender, region1))

# #                 # same gender, different nationality
# #                 for name2 in group2_same:
# #                     if region1 != region2 and name1 != name2:
# #                         name_pairs.append((name1, name2, gender, region1, gender, region2))

# #                 # different gender, different nationality
# #                 for name2 in group2_opposite:
# #                     if region1 != region2 and name1 != name2:
# #                         name_pairs.append((name1, name2, gender, region1, opposite_gender, region2))

# #     return name_pairs[:3]


# def generate_name_pairs(names_data, gender):
#     name_pairs = []

#     # Get gender variations
#     opposite_gender = "male" if gender == "female" else "female"

#     seen = {
#         "same_gender_same_nation": False,
#         "diff_gender_same_nation": False,
#         "same_gender_diff_nation": False,
#         "diff_gender_diff_nation": False,
#     }

#     for region1, group1 in names_data.items():
#         # group1_nested = group1.get(gender.capitalize(), [])
#         # group1_opposite = group1.get(opposite_gender.capitalize(), [])
#         #testing
#         group1_nested = group1.get(gender.capitalize(), [])[:1]
#         group1_opposite = group1.get(opposite_gender.capitalize(), [])[:1]
#         #testing


#         for name1 in group1_nested:
#             for region2, group2 in names_data.items():
#                 # group2_same = group2.get(gender.capitalize(), [])
#                 # group2_opposite = group2.get(opposite_gender.capitalize(), [])
#                 #testing
#                 group2_same = group2.get(gender.capitalize(), [])[:1]
#                 group2_opposite = group2.get(opposite_gender.capitalize(), [])[:1]
#                 #testing

#                 # same gender + same nationality
#                 if not seen["same_gender_same_nation"]:
#                     for name2 in group1_nested:
#                         if name1 != name2:
#                             name_pairs.append((name1, name2, gender, region1, gender, region1))
#                             seen["same_gender_same_nation"] = True
#                             break

#                 # different gender, same nationality
#                 if not seen["diff_gender_same_nation"]:
#                     for name2 in group1_opposite:
#                         if name1 != name2:
#                             name_pairs.append((name1, name2, gender, region1, opposite_gender, region1))
#                             seen["diff_gender_same_nation"] = True
#                             break

#                 # same gender, different nationality
#                 if not seen["same_gender_diff_nation"] and region1 != region2:
#                     for name2 in group2_same:
#                         if name1 != name2:
#                             name_pairs.append((name1, name2, gender, region1, gender, region2))
#                             seen["same_gender_diff_nation"] = True
#                             break

#                 # different gender, different nationality
#                 if not seen["diff_gender_diff_nation"] and region1 != region2:
#                     for name2 in group2_opposite:
#                         if name1 != name2:
#                             name_pairs.append((name1, name2, gender, region1, opposite_gender, region2))
#                             seen["diff_gender_diff_nation"] = True
#                             break

#             if all(seen.values()):
#                 return name_pairs

#     return name_pairs


# def generate_serialized_entries(input_data, names_data, selected_domains=None, selected_lengths=None):
#     serialized_entries = []

#     for gender in ["male", "female"]:
#         if gender not in input_data:
#             continue
#         for domain, length_group in input_data[gender].items():
#             if selected_domains and domain not in selected_domains:
#                 continue
#             for length, prompts in length_group.items():
#                 if selected_lengths and length not in selected_lengths:
#                     continue
#                 for idx, prompt_group in enumerate(prompts):
#                     for prompt in prompt_group:
#                         #testing
#                         # Limit to scenario_id 1a–1l
#                         scenario_id = prompt.get("scenario_id", "")
#                         if not scenario_id.startswith("1") or scenario_id not in [f"1{c}" for c in "abcdefghijkl"]:
#                             continue

#                         # Filter by attribution type for testing
#                         actor_type = prompt.get("actor_attribution_type", "")
#                         observer_type = prompt.get("observer_attribution_type", "")
#                         if actor_type not in ["Effort", "Ability"] or observer_type not in ["Difficulty", "Luck"]:
#                             continue
#                         #testing
#                         for region, group in names_data.items():
#                             if isinstance(group, dict) and all(isinstance(v, dict) for v in group.values()):  # Proper nested
#                                 for sub_category, sub_group in group.items():
#                                     names_list = sub_group.get(gender.capitalize(), [])
#                                     name_pairs = generate_name_pairs(names_data, gender)
#                                     for name_x, name_y, gender_x, nat_x, gender_y, nat_y in name_pairs:
#                                         entry = prompt.copy()
#                                         # for field in ["scenario", "dialogue", "question"]:
#                                         #     if field in entry:
#                                         #         entry[field] = entry[field].replace("{X}", name_x).replace("{Y}", name_y)
#                                         # Replace placeholders
#                                         for field in ["scenario", "dialogue", "question"]:
#                                             if field in entry:
#                                                 entry[field] = entry[field].replace("{X}", name_x).replace("{Y}", name_y)

#                                         # Extract what X and Y said from dialogue (for option1 and option2)
#                                         dialogue_text = entry.get("dialogue", "")
#                                         quotes = dialogue_text.split("'")

#                                         if len(quotes) >= 4:
#                                             x_sentence = quotes[1].replace("I", name_x)
#                                             y_sentence = quotes[3]
#                                             entry["option1"] = x_sentence
#                                             entry["option2"] = y_sentence
#                                         else:
#                                             entry["option1"] = f"What {name_x} said"
#                                             entry["option2"] = f"What {name_y} said"
#                                         entry["gender_X"] = gender_x
#                                         entry["gender_Y"] = gender_y
#                                         entry["nationality_X"] = nat_x
#                                         entry["nationality_Y"] = nat_y
#                                         entry["name_X"] = name_x
#                                         entry["name_Y"] = name_y
#                                         entry["domain"] = domain
#                                         entry["length"] = length
#                                         entry["set_id"] = f"{MODE}_{gender}_{domain}_{length}_set{idx}_{name_x}_{name_y}"
#                                         serialized_entries.append(entry)

#                             else:  # Flat (e.g., African Names, American Names)
#                                 name_pairs = generate_name_pairs(names_data, gender)
#                                 for name_x, name_y, gender_x, nat_x, gender_y, nat_y in name_pairs:
#                                     entry = prompt.copy()
#                                     # for field in ["scenario", "dialogue", "question"]:
#                                     #     if field in entry:
#                                     #         entry[field] = entry[field].replace("{X}", name_x).replace("{Y}", name_y)
#                                     # Replace placeholders
#                                     for field in ["scenario", "dialogue", "question"]:
#                                         if field in entry:
#                                             entry[field] = entry[field].replace("{X}", name_x).replace("{Y}", name_y)

#                                     # Extract what X and Y said from dialogue (for option1 and option2)
#                                     dialogue_text = entry.get("dialogue", "")
#                                     quotes = dialogue_text.split("'")

#                                     if len(quotes) >= 4:
#                                         x_sentence = quotes[1].replace("I", name_x)
#                                         y_sentence = quotes[3]
#                                         entry["option1"] = x_sentence
#                                         entry["option2"] = y_sentence
#                                     else:
#                                         entry["option1"] = f"What {name_x} said"
#                                         entry["option2"] = f"What {name_y} said"
#                                     entry["gender_X"] = gender_x
#                                     entry["gender_Y"] = gender_y
#                                     entry["nationality_X"] = nat_x
#                                     entry["nationality_Y"] = nat_y
#                                     entry["name_X"] = name_x
#                                     entry["name_Y"] = name_y
#                                     entry["domain"] = domain
#                                     entry["length"] = length
#                                     entry["set_id"] = f"{MODE}_{gender}_{domain}_{length}_set{idx}_{name_x}_{name_y}"
#                                     serialized_entries.append(entry)
#     return serialized_entries

# # Generate entries with names inserted
# all_entries = generate_serialized_entries(nested_input_data, names_data, selected_domains, selected_lengths)
# # data = all_entries[:2]  # Limit for testing

# print(f"Generated {len(all_entries)} entries.") 

# if all_entries:
#     print("Sample Entry:", all_entries[0])  # Print the first entry for debugging


# # Filter out already processed entries if resuming
# if os.path.exists(PARTIAL_CSV_FILE):
#     print(f"🔁 Resuming from partial file: {PARTIAL_CSV_FILE}")
#     prev_df = pd.read_csv(PARTIAL_CSV_FILE)

#     # Convert stringified dicts back to dicts
#     if "option_probs" in prev_df.columns:
#         prev_df["option_probs"] = prev_df["option_probs"].apply(
#             lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x
#         )

#     if "shuffled_options" in prev_df.columns:
#         prev_df["shuffled_options"] = prev_df["shuffled_options"].apply(
#             lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x
#         )

#     processed_set_ids = set(prev_df["set_id"] + "_" + prev_df["name_X"] + "_" + prev_df["name_Y"])
#     all_entries = [e for e in all_entries if e["set_id"] + "_" + e["name_X"] + "_" + e["name_Y"] not in processed_set_ids]
#     all_runs_results = [prev_df]
# else:
#     all_runs_results = []


# # Save combined entries to a JSON file
# with open(OUTPUT_JSON_FILE, "w") as f:
#     json.dump(all_entries, f, indent=4)


# def get_completion_with_cappr(model, tokenizer, entry_data):
#     try:
#         scenario = entry_data["scenario"]
#         dialogue = entry_data["dialogue"]
#         question = entry_data["question"]
#         options = entry_data["shuffled_options"]

#         prompt = (
#             f"Below is a scenario and dialogue followed by a multiple-choice question."
#             f"Respond by choosing one of the provided options as the reason to the question.\n\n"
#             f"### Scenario:\n{scenario}\n\n"
#             f"### Dialogue:\n{dialogue}\n\n"
#             f"### Question:\n{question}\n\n"
#             f"Options:\n"
#             f"A. {options[0]}\n"
#             f"B. {options[1]}\n"
#             f"### Response:\n"
#         )

#         # Use cappr to calculate probabilities
#         pred_probs = predict_proba(
#             prompt,
#             completions=options,
#             model_and_tokenizer=(model, tokenizer),
#             end_of_prompt="",
#             batch_size=1,
#         )
#         pred_probs_rounded = pred_probs.round(2)

#         # Map probabilities back to option letters
#         option_probs = {chr(65 + i): prob for i, prob in enumerate(pred_probs_rounded)}

#         # Determine the most probable option
#         chosen_idx = pred_probs.argmax()
#         chosen_letter = chr(65 + chosen_idx)  # Convert index to A, B, C, D

#         chosen_sentence = options[chosen_idx]

#         return {
#             "scenario": scenario,
#             "dialogue": dialogue,
#             "question": question,
#             "chosen_letter": chosen_letter,
#             "chosen_key": entry_data["shuffled_option_mapping"][chosen_letter],
#             "chosen_sentence": chosen_sentence,
#             "shuffled_options": {key: entry_data["shuffled_option_mapping"][key] for key in entry_data["shuffled_option_mapping"]},
#             "option_probs": option_probs,
#         }

#     except Exception as e:
#         print(f"Error during processing: {e}")
#         return None


# def run_experiment(data, model, tokenizer, run_number, seed):
#     total_results = []
    
#     progress_bar = tqdm(total=len(data), desc="Processing entries")
#     for batch_start in range(0, len(data), BATCH_SIZE):
#         batch = data[batch_start: batch_start + BATCH_SIZE]
#         processed_entries = []

#         for idx, entry in enumerate(batch):
#             original_options = OrderedDict([("option1", entry["option1"]), ("option2", entry["option2"])])

#             random.seed(seed + batch_start + idx)
#             shuffled_items = list(original_options.items())
#             random.shuffle(shuffled_items)
#             shuffled_options = [item[1] for item in shuffled_items]
#             shuffled_option_mapping = {chr(65 + i): key for i, (key, _) in enumerate(shuffled_items)}

#             processed_entries.append({
#                 "index": batch_start + idx,
#                 "original_options": dict(original_options),
#                 "shuffled_options": shuffled_options,
#                 "shuffled_option_mapping": shuffled_option_mapping,
#                 "gender_X": entry["gender_X"],
#                 "gender_Y": entry["gender_Y"],
#                 "nationality_X": entry["nationality_X"],
#                 "nationality_Y": entry["nationality_Y"],
#                 "name_X": entry["name_X"],
#                 "name_Y": entry["name_Y"],
#                 "set_id": entry["set_id"],
#                 "scenario": entry["scenario"],               # ✅ Add this
#                 "dialogue": entry["dialogue"],               # ✅ Add this
#                 "question": entry["question"]                # ✅ Add this
#             })

#         # works for llama
#         # Run cappr in parallel
#         with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
#             futures = {
#                 executor.submit(get_completion_with_cappr, model, tokenizer, entry_data): entry_data["index"]
#                 for entry_data in processed_entries
#             }

#             results = []
#             for future in as_completed(futures):
#                 result = future.result()
#                 if result is not None:
#                     results.append((futures[future], result))
#                 progress_bar.update(1)
#         # works for llama


#         # # works for aya
#         # results = []
#         # for entry_data in processed_entries:
#         #     result = get_completion_with_cappr(model, tokenizer, entry_data)
#         #     if result is not None:
#         #         results.append((entry_data["index"], result))
#         #     progress_bar.update(1)
#         # # works for aya


#         results.sort(key=lambda x: x[0])
#         sorted_results = [res[1] for res in results]
#         result_df = pd.DataFrame(sorted_results)

#         # Add metadata
#         result_df["run"] = run_number
#         result_df["seed"] = seed
#         result_df["set_id"] = [entry["set_id"] for entry in batch]
#         result_df["domain"] = [entry["domain"] for entry in batch]
#         result_df["length"] = [entry["length"] for entry in batch]
#         result_df["gender_X"] = [entry["gender_X"] for entry in batch]
#         result_df["gender_Y"] = [entry["gender_Y"] for entry in batch]
#         result_df["nationality_X"] = [entry["nationality_X"] for entry in batch]
#         result_df["nationality_Y"] = [entry["nationality_Y"] for entry in batch]
#         result_df["name_X"] = [entry["name_X"] for entry in batch]
#         result_df["name_Y"] = [entry["name_Y"] for entry in batch]


#         # Expand option_probs
#         expanded_probs_df = pd.DataFrame(result_df.apply(expand_option_probs, axis=1).tolist())
#         result_df = pd.concat([result_df, expanded_probs_df], axis=1)


#         # ✅ Save this batch
#         result_df.to_csv(PARTIAL_CSV_FILE, mode='a', index=False, header=not os.path.exists(PARTIAL_CSV_FILE))

#         total_results.append(result_df)

#     progress_bar.close()
#     if not total_results:
#         print("✅ Nothing left to process. All entries already completed.")
#         return pd.DataFrame()
#     return pd.concat(total_results, ignore_index=True)

# # Load JSON data
# with open(OUTPUT_JSON_FILE, "r") as f:
#     data = json.load(f)
# # data = data[:2]

# # Expand option_probs into individual columns
# def expand_option_probs(row):
#     option_columns = {}
#     for option_letter, prob in row["option_probs"].items():
#         original_option_name = row["shuffled_options"][option_letter]
#         option_columns[original_option_name] = prob
#     return option_columns


# all_runs_results = []
# for run in tqdm(range(args.runs), total=args.runs, desc="Runs:"):
#     new_seed = random.randint(0, 100000)
#     set_seed(new_seed)
#     print(f"Run {run + 1}/{args.runs} with seed {new_seed}")
#     result_df = run_experiment(data, model, tokenizer, run + 1, new_seed)
#     #insert here
#     # ✅ Correct: work with result_df

# # Reload final partial CSV instead of relying on in-memory all_runs_results
# if os.path.exists(PARTIAL_CSV_FILE):
#     final_result_df = pd.read_csv(PARTIAL_CSV_FILE)

#     # Optional: re-expand dicts if needed downstream
#     final_result_df["option_probs"] = final_result_df["option_probs"].apply(
#         lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x
#     )
#     final_result_df["shuffled_options"] = final_result_df["shuffled_options"].apply(
#         lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else x
#     )
# else:
#     final_result_df = pd.concat(all_runs_results)

# # Save the final result to CSV
# output_file = f"{output_dir}/actor_observer_{MODE}_{args.model}_{length_str}_{args.runs}_runs.csv"
# final_result_df.to_csv(output_file, index=False)
# print(f"Results saved to {output_file}")



