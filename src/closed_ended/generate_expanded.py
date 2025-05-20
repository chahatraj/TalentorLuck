import json
import argparse
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import os
import re
import ast

# --------- Argument parser ---------
parser = argparse.ArgumentParser(description="Generate expanded JSON entries.")
parser.add_argument("--mode", type=str, choices=["success", "failure", "both_success", "both_failure", "success_failure"], default="success", help="Mode to run: success or failure")
parser.add_argument("--dimension", type=str, choices=["nationality", "race", "religion"], default="nationality", help="Select dimension")
parser.add_argument("--domains", nargs="+", default=None, help="Domains to process (e.g., education workplace or 'all')")
parser.add_argument("--lengths", nargs="+", default=["short"], help="Lengths to process...")
parser.add_argument("--limit", type=int, default=None, help="Limit number of entries")
parser.add_argument("--input_type", type=str, choices=["single_actor", "actor_actor", "actor_observer"], default="single_actor", help="Input type")
parser.add_argument("--name_limit", type=int, default=None, help="Limit number of names to use per gender/dimension")
args = parser.parse_args()

# --------- File paths ---------
MODE = args.mode
SUCCESS_INPUT = f"../../data/closed_ended/single_actor/success.json"
FAILURE_INPUT = f"../../data/closed_ended/single_actor/failure.json"
BOTH_SUCCESS_INPUT = f"../../data/closed_ended/actor_actor/both_success.json"
BOTH_FAILURE_INPUT = f"../../data/closed_ended/actor_actor/both_failure.json"
SUCCESS_FAILURE_INPUT = f"../../data/closed_ended/actor_actor/success_failure.json"
ACTOR_OBSERVER_SUCCESS_INPUT = f"../../data/closed_ended/actor_observer/success.json"
ACTOR_OBSERVER_FAILURE_INPUT = f"../../data/closed_ended/actor_observer/failure.json"


NAMES_FILE = f"../../data/names/{args.dimension}.json"

length_str = "all" if args.lengths is None else "_".join(args.lengths)
domain_str = "all" if args.domains is None else "_".join(args.domains)
dimension_str = args.dimension
OUTPUT_JSON_FILE = f"../../data/closed_ended/{args.input_type}/{MODE}_{dimension_str}_{domain_str}_limited.json"
os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)


if args.input_type == "single_actor":
    INPUT_FILE = SUCCESS_INPUT if MODE == "success" else FAILURE_INPUT
elif args.input_type == "actor_actor":
    if MODE == "both_success":
        INPUT_FILE = BOTH_SUCCESS_INPUT
    elif MODE == "both_failure":
        INPUT_FILE = BOTH_FAILURE_INPUT
    elif MODE == "success_failure":
        INPUT_FILE = SUCCESS_FAILURE_INPUT
    else:
        raise ValueError(f"Invalid mode '{MODE}' for actor_actor input")
elif args.input_type == "actor_observer":
    INPUT_FILE = ACTOR_OBSERVER_SUCCESS_INPUT if MODE == "success" else ACTOR_OBSERVER_FAILURE_INPUT
else:
    raise ValueError(f"Unknown input_type: {args.input_type}")



# --------- Load input data ---------
with open(INPUT_FILE, "r") as f:
    nested_input_data = json.load(f)
with open(NAMES_FILE, "r") as f:
    names_data = json.load(f)

# --------- Collect domains and lengths ---------
all_domains = set()
all_lengths = set()

if args.input_type == "single_actor":
    for gender_data in nested_input_data.values():
        for domain, lengths in gender_data.items():
            all_domains.add(domain)
            all_lengths.update(lengths.keys())
elif args.input_type == "actor_actor": 
    for domain, lengths in nested_input_data.items():
        all_domains.add(domain)
        all_lengths.update(lengths.keys())
elif args.input_type == "actor_observer":
    for gender_data in nested_input_data.values():
        for domain, lengths in gender_data.items():
            all_domains.add(domain)
            all_lengths.update(lengths.keys())


selected_domains = all_domains if args.domains == ["all"] else set(args.domains) if args.domains else all_domains
selected_lengths = all_lengths if args.lengths == ["all"] else set(args.lengths) if args.lengths else all_lengths

# --------- Generate serialized entries ---------
def generate_serialized_entries(input_data, names_data, selected_domains, selected_lengths):
    serialized_entries = []
    if not all("male_names" in v and "female_names" in v for v in names_data.values()):
        raise ValueError("Invalid names JSON: each dimension must have 'male_names' and 'female_names'.")

    # for gender in ["male", "female"]:
    for gender in tqdm(["male", "female"], desc="Gender"):
        gender_key = f"{gender}_names"
        # for domain, length_group in input_data[gender].items():
        for domain, length_group in tqdm(input_data[gender].items(), desc=f"{gender} Domains"):
            if domain not in selected_domains:
                continue
            for length, prompts in length_group.items():
                if length not in selected_lengths:
                    continue
                for idx, prompt in enumerate(prompts):
                    for dimension_label, names_dict in names_data.items():
                        names_list = names_dict.get(gender_key, [])
                        if args.name_limit:
                            names_list = names_list[:args.name_limit]
                        for name in names_list:
                            entry = prompt.copy()
                            entry["initial_prompt"] = entry["initial_prompt"].replace("{X}", name).replace("{dimension}", dimension_label)
                            entry["gender"] = gender
                            entry["domain"] = domain
                            entry["length"] = length
                            entry["name"] = name
                            entry["dimension"] = dimension_label
                            entry["set_id"] = f"{MODE}_{gender}_{domain}_{length}_set{idx}"
                            serialized_entries.append(entry)
    return serialized_entries


# def generate_serialized_entries_actor_actor(input_data, names_data, selected_domains, selected_lengths):
#     serialized_entries = []
#     dimensions = list(names_data.keys())

#     # for domain, length_group in input_data.items():
#     for domain, length_group in tqdm(input_data.items(), desc="Domains"):
#         if domain not in selected_domains:
#             continue
#         # for length, prompts in length_group.items():
#         for length, prompts in tqdm(length_group.items(), desc=f"{domain} Lengths"):
#             if length not in selected_lengths:
#                 continue
#             for idx, prompt in enumerate(prompts):
#                 for gender_pair, options in prompt["gender_pairs"].items():
#                     gender1, gender2 = gender_pair.split("-")

#                     for dimension1 in dimensions:
#                         for dimension2 in dimensions:
#                             names1_dim = names_data[dimension1].get(f"{gender1}_names", [])
#                             names2_dim = names_data[dimension2].get(f"{gender2}_names", [])
#                             if args.name_limit:
#                                 names1_dim = names1_dim[:args.name_limit]
#                                 names2_dim = names2_dim[:args.name_limit]

#                             for name1 in names1_dim:
#                                 for name2 in names2_dim:
#                                     if name1 == name2:
#                                         continue  # Avoid using same name for X and Y

#                                     entry = prompt.copy()

#                                     entry["initial_prompt"] = entry["initial_prompt"] \
#                                         .replace("{X}", name1) \
#                                         .replace("{Y}", name2) \
#                                         .replace("{dimension_1}", dimension1) \
#                                         .replace("{dimension_2}", dimension2)

#                                     entry["gender_pair"] = gender_pair
#                                     entry["domain"] = domain
#                                     entry["length"] = length
#                                     entry["name1"] = name1
#                                     entry["name2"] = name2
#                                     entry["dimension1"] = dimension1
#                                     entry["dimension2"] = dimension2
#                                     entry["set_id"] = f"{MODE}_{gender_pair}_{domain}_{length}_set{idx}"

#                                     # Flatten options for both success/failure or both success
#                                     entry.pop("gender_pairs")
#                                     entry.update(options)

#                                     serialized_entries.append(entry)
#     return serialized_entries

def generate_serialized_entries_actor_actor(input_data, names_data, selected_domains, selected_lengths):
    serialized_entries = []
    dimensions = list(names_data.keys())
    pair_counter = defaultdict(int)  # To count pairs per (dim1, dim2, domain, gender_pair, set_id)

    for domain, length_group in tqdm(input_data.items(), desc="Domains"):
        if domain not in selected_domains:
            continue
        for length, prompts in tqdm(length_group.items(), desc=f"{domain} Lengths"):
            if length not in selected_lengths:
                continue
            for idx, prompt in enumerate(prompts):
                for gender_pair, options in prompt["gender_pairs"].items():
                    gender1, gender2 = gender_pair.split("-")

                    for dimension1 in dimensions:
                        for dimension2 in dimensions:
                            names1_dim = names_data[dimension1].get(f"{gender1}_names", [])
                            names2_dim = names_data[dimension2].get(f"{gender2}_names", [])
                            if args.name_limit:
                                names1_dim = names1_dim[:args.name_limit]
                                names2_dim = names2_dim[:args.name_limit]

                            # Track how many pairs we’ve added for this combination
                            pair_key = f"{dimension1}_{dimension2}_{domain}_{gender_pair}_{idx}"

                            for name1 in names1_dim:
                                for name2 in names2_dim:
                                    if name1 == name2:
                                        continue  # Avoid using same name for X and Y

                                    if pair_counter[pair_key] >= 5:
                                        break  # Stop adding more pairs

                                    entry = prompt.copy()

                                    entry["initial_prompt"] = entry["initial_prompt"] \
                                        .replace("{X}", name1) \
                                        .replace("{Y}", name2) \
                                        .replace("{dimension_1}", dimension1) \
                                        .replace("{dimension_2}", dimension2)

                                    entry["gender_pair"] = gender_pair
                                    entry["domain"] = domain
                                    entry["length"] = length
                                    entry["name1"] = name1
                                    entry["name2"] = name2
                                    entry["dimension1"] = dimension1
                                    entry["dimension2"] = dimension2
                                    entry["set_id"] = f"{MODE}_{gender_pair}_{domain}_{length}_set{idx}"

                                    # Flatten options for both success/failure or both success
                                    entry.pop("gender_pairs")
                                    entry.update(options)

                                    serialized_entries.append(entry)
                                    pair_counter[pair_key] += 1

                                if pair_counter[pair_key] >= 5:
                                    break  # Stop outer loop as well

    return serialized_entries



# def generate_serialized_entries_actor_observer(input_data, names_data, selected_domains, selected_lengths):
#     serialized_entries = []
#     dimensions = list(names_data.keys())

#     for gender, gender_data in tqdm(input_data.items(), desc="Gender"):
#         for domain, length_group in tqdm(gender_data.items(), desc=f"{gender} Domains"):
#             if domain not in selected_domains:
#                 continue
#             for length, prompt_list in length_group.items():
#                 if length not in selected_lengths:
#                     continue
#                 for prompt_group in prompt_list:
#                     for y_opt_key, prompt in prompt_group.items():
#                         for dimension1 in dimensions:
#                             for dimension2 in dimensions:
#                                 names1_dim = names_data[dimension1].get(f"{gender}_names", [])
#                                 names2_dim = names_data[dimension2].get(f"{gender}_names", [])
#                                 if args.name_limit:
#                                     names1_dim = names1_dim[:args.name_limit]
#                                     names2_dim = names2_dim[:args.name_limit]
#                                 for name1 in names1_dim:
#                                     for name2 in names2_dim:
#                                         if name1 == name2:
#                                             continue

#                                         entry = prompt.copy()
#                                         entry["initial_prompt"] = entry["initial_prompt"] \
#                                             .replace("{X}", name1) \
#                                             .replace("{Y}", name2) \
#                                             .replace("{dimension_1}", dimension1) \
#                                             .replace("{dimension_2}", dimension2)
#                                         entry["gender"] = gender
#                                         entry["domain"] = domain
#                                         entry["length"] = length
#                                         entry["name1"] = name1
#                                         entry["name2"] = name2
#                                         entry["dimension1"] = dimension1
#                                         entry["dimension2"] = dimension2
#                                         entry["y_opt_key"] = y_opt_key

#                                         serialized_entries.append(entry)
#     return serialized_entries


def generate_serialized_entries_actor_observer(input_data, names_data, selected_domains, selected_lengths):
    serialized_entries = []
    dimensions = list(names_data.keys())
    pair_counter = defaultdict(int)  # Track count per combination

    for gender, gender_data in tqdm(input_data.items(), desc="Gender"):
        for domain, length_group in tqdm(gender_data.items(), desc=f"{gender} Domains"):
            if domain not in selected_domains:
                continue
            for length, prompt_list in length_group.items():
                if length not in selected_lengths:
                    continue
                for prompt_group in prompt_list:
                    for y_opt_key, prompt in prompt_group.items():
                        for dimension1 in dimensions:
                            for dimension2 in dimensions:
                                names1_dim = names_data[dimension1].get(f"{gender}_names", [])
                                names2_dim = names_data[dimension2].get(f"{gender}_names", [])
                                if args.name_limit:
                                    names1_dim = names1_dim[:args.name_limit]
                                    names2_dim = names2_dim[:args.name_limit]

                                # Create counter key
                                pair_key = f"{gender}_{domain}_{length}_{y_opt_key}_{dimension1}_{dimension2}"

                                for name1 in names1_dim:
                                    for name2 in names2_dim:
                                        if name1 == name2:
                                            continue

                                        if pair_counter[pair_key] >= 5:
                                            break  # Stop adding more pairs

                                        entry = prompt.copy()
                                        entry["initial_prompt"] = entry["initial_prompt"] \
                                            .replace("{X}", name1) \
                                            .replace("{Y}", name2) \
                                            .replace("{dimension_1}", dimension1) \
                                            .replace("{dimension_2}", dimension2)
                                        entry["gender"] = gender
                                        entry["domain"] = domain
                                        entry["length"] = length
                                        entry["name1"] = name1
                                        entry["name2"] = name2
                                        entry["dimension1"] = dimension1
                                        entry["dimension2"] = dimension2
                                        entry["y_opt_key"] = y_opt_key

                                        serialized_entries.append(entry)
                                        pair_counter[pair_key] += 1

                                    if pair_counter[pair_key] >= 5:
                                        break  # Stop outer loop

    return serialized_entries



# --------- Run generation and save ---------
if args.input_type == "single_actor":
    all_entries = generate_serialized_entries(nested_input_data, names_data, selected_domains, selected_lengths)
elif args.input_type == "actor_actor":
    all_entries = generate_serialized_entries_actor_actor(nested_input_data, names_data, selected_domains, selected_lengths)
elif args.input_type == "actor_observer":
    all_entries = generate_serialized_entries_actor_observer(nested_input_data, names_data, selected_domains, selected_lengths)

if args.limit:
    all_entries = all_entries[:args.limit]


with open(OUTPUT_JSON_FILE, "w") as f:
    json.dump(all_entries, f, ensure_ascii=False, indent=4)
    # json.dump(all_entries, f)
print(f"✅ Saved {len(all_entries)} entries to {OUTPUT_JSON_FILE}")

