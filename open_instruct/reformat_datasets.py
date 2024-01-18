#!/usr/bin/env python
# coding=utf-8
'''
This script is used to reformat the downloaded datasets into the format that can be used by the model.
Here we use jsonl for the converted data. Each line in the jsonl file is a json object formatted as follows:
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
'''

import json
import random
import re
import os
import pandas as pd
import argparse
from datasets import load_dataset


def convert_evol_codealpaca_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset('theblackcat102/evol-codealpaca-v1')
    output_path = os.path.join(output_dir, "evol_codealpaca_v1.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(ds['train']):
            # split example["input"] by [|Human|] and [|AI|]
            messages = []
            
            messages.append({
                "role": "user",
                "content": example['instruction']
            })
            messages.append({
                "role": "assistant",
                "content": example['output']
            })
            fout.write(json.dumps({
                "dataset": "evol_codealpaca_v1",
                "id": f"evol_codealpaca_v1_{idx}",
                "messages": messages
            }) + "\n")


def convert_meta_math_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset('meta-math/MetaMathQA')
    output_path = os.path.join(output_dir, "MetaMathQA.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(ds['train']):
            # split example["input"] by [|Human|] and [|AI|]
            messages = []
            
            messages.append({
                "role": "user",
                "content": example['query']
            })
            messages.append({
                "role": "assistant",
                "content": example['response']
            })
            fout.write(json.dumps({
                "dataset": "MetaMathQA",
                "id": f"MetaMathQA_{idx}",
                "messages": messages
            }) + "\n")



def convert_WizardLM_evol_instruct_V2_196k_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset('WizardLM/WizardLM_evol_instruct_V2_196k')
    output_path = os.path.join(output_dir, "WizardLM_evol_instruct_V2_196k.jsonl")
    with open(output_path, "w") as fout:
        invalid_cnt = 0
        for idx, example in enumerate(ds['train']):
            messages = []
            valid = True
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["value"]
                    })
                elif message["from"] == "gpt" or message["from"] == "chatgpt":
                    messages.append({
                        "role": "assistant",
                        "content": message["value"]
                    })
                elif message["from"] == "system":
                    valid = False
                    invalid_cnt += 1
                    break
                elif message["from"] == "bing":
                    valid = False
                    invalid_cnt += 1
                    break
                else:
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if messages and valid:
                fout.write(json.dumps({
                    "dataset": "WizardLM_evol_instruct_V2_196k",
                    "id": f"WizardLM_evol_instruct_V2_196k_{idx}",
                    "messages": messages
                }) + "\n")
        print(f"# of invalid examples in WizardLM_evol_instruct_V2_196k data: {invalid_cnt}")


def convert_SlimOrca_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset('Open-Orca/SlimOrca')
    output_path = os.path.join(output_dir, "SlimOrca.jsonl")
    with open(output_path, "w") as fout:
        invalid_cnt = 0
        for idx, example in enumerate(ds['train']):
            messages = []
            valid = True
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["value"]
                    })
                elif message["from"] == "gpt" or message["from"] == "chatgpt":
                    messages.append({
                        "role": "assistant",
                        "content": message["value"]
                    })
                elif message["from"] == "system":
                    messages.append({
                        "role": "system",
                        "content": message["value"]
                    })
                else:
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if messages and valid:
                fout.write(json.dumps({
                    "dataset": "SlimOrca",
                    "id": f"SlimOrca_{idx}",
                    "messages": messages
                }) + "\n")
        print(f"# of invalid examples in SlimOrca data: {invalid_cnt}")



if __name__ == "__main__":
    # all supported datasets    
    supported_datasets = []
    all_funcs = [func_name for func_name in globals() if callable(globals()[func_name])]
    for func_name in all_funcs:
        if re.match(r"convert_.+_data", func_name):
            supported_datasets.append(func_name[8:-5])

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        default="data/downloads"
    )
    arg_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed"
    )
    arg_parser.add_argument(
        "--dataset", 
        type=str, 
        nargs="+"
    )
    arg_parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    args = arg_parser.parse_args()
    random.seed(args.seed)

    for dataset in args.dataset:
        print(f"Processing {dataset} data with default configurations...")
        globals()[f"convert_{dataset}_data"](os.path.join(args.raw_data_dir, dataset), os.path.join(args.output_dir, dataset))