import argparse
import json
import os
import random
from collections import defaultdict

import torch
import vllm
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval.predict import llama2_prompting_template
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
)


@torch.no_grad()
def score_generations(generations, batch_size=32):
    classifications = []
    for i in trange(0, len(generations), batch_size):
        batch = generations[i: i + batch_size]
        inputs = toxigenRobertaTokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda")
        outputs = toxigenRobertaModel(**inputs)
        classifications.extend(outputs.logits.argmax(dim=-1).tolist())
    return classifications


@torch.no_grad()
def eval_vllm_model(args, model, examples, save_path=None):
    if save_path:
        fout = open(save_path, "w")

    model = vllm.LLM(model=args.model_name_or_path)
    prompts = [it["text"] for it in examples]
    if args.use_llama_chat_format:
        prompts = [llama2_prompting_template.format(prompt=x) for x in prompts]
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=512,
    )
    outputs = model.generate(prompts, sampling_params)
    outputs = [it.outputs[0].text for it in outputs]
    classifications = score_generations(outputs)

    all_groups_scores = defaultdict(list)
    for example, output, classification in zip(examples, outputs, classifications):
        example["prediction"] = output
        example["is_toxic"] = classification
        for group in example["target_groups"]:
            all_groups_scores[group].append(classification)
        if save_path:
            fout.write(json.dumps(example) + "\n")

    performance = {}
    # calculate score by targeted group
    for group in all_groups_scores:
        performance[group] = sum(all_groups_scores[group]) / len(
            all_groups_scores[group]
        )
    # overall perf
    performance["overall"] = sum(classifications) / len(classifications)

    return performance


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, examples, save_path=None):
    if save_path:
        fout = open(save_path, "w")

    prompts = []
    for example in examples:
        if args.use_chat_format:
            prompt = "<|user|>\n" + example["text"] + "\n<|assistant|>\nA:"
        else:
            prompt = example["text"]
        prompts.append(prompt)

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
    )

    classifications = score_generations(outputs)

    all_groups_scores = defaultdict(list)
    for example, output, classification in zip(examples, outputs, classifications):
        example["prediction"] = output
        example["is_toxic"] = classification
        for group in example["target_groups"]:
            all_groups_scores[group].append(classification)
        if save_path:
            fout.write(json.dumps(example) + "\n")

    performance = {}
    # calculate score by targeted group
    for group in all_groups_scores:
        performance[group] = sum(all_groups_scores[group]) / len(
            all_groups_scores[group]
        )
    # overall perf
    performance["overall"] = sum(classifications) / len(classifications)

    return performance


def eval_openai_chat_engine(args, examples, save_path=None):
    instances = []
    for i, example in enumerate(examples):
        prompt = example["text"]
        instances.append(
            {
                "id": example["id"] if "id" in example else i,
                "prompt": prompt,
            }
        )

    if save_path:
        openai_result_save_path = os.path.join(
            os.path.dirname(save_path),
            os.path.basename(save_path).split(".")[0] + "_openai_results.jsonl",
        )

    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=openai_result_save_path if save_path else None,
    )

    outputs = [result["output"] for result in results]

    classifications = score_generations(outputs)

    if save_path:
        fout = open(save_path, "w")

    all_groups_scores = defaultdict(list)
    for example, output, classification in zip(examples, outputs, classifications):
        example["prediction"] = output
        example["is_toxic"] = classification
        for group in example["target_groups"]:
            all_groups_scores[group].append(classification)
        if save_path:
            fout.write(json.dumps(example) + "\n")

    performance = {}
    # calculate score by targeted group
    for group in all_groups_scores:
        performance[group] = sum(all_groups_scores[group]) / len(
            all_groups_scores[group]
        )
    # overall perf
    performance["overall"] = sum(classifications) / len(classifications)

    return performance


def main(args):
    random.seed(42)

    all_prompts = json.load(open(os.path.join(args.data_dir, "toxiGen.json"), "r"))

    # we assume running on a gpu here.
    toxigenRobertaTokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta")
    toxigenRobertaModel = AutoModelForSequenceClassification.from_pretrained(
        "tomh/toxigen_roberta"
    ).cuda()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    if args.model_name_or_path:
        if args.use_vllm:
            results = eval_vllm_model(
                args,
                args.model_name_or_path,
                all_prompts,
                save_path=os.path.join(args.save_dir, "predictions", "toxigen.jsonl"),
            )
        else:
            print("Loading model and tokenizer...")
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0"
                if torch.cuda.device_count() > 1
                else "auto",
                gptq_model=args.gptq,
            )
            results = eval_hf_model(
                args,
                model,
                tokenizer,
                all_prompts,
                save_path=os.path.join(args.save_dir, "predictions", "toxigen.jsonl"),
            )
    else:
        results = eval_openai_chat_engine(
            args,
            all_prompts,
            save_path=os.path.join(args.save_dir, "predictions", "toxigen.jsonl"),
        )

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        print(f"Average Toxicity: {results['overall']}")
        for group in results:
            if group != "overall":
                print(f"Average Toxicity for {group}: {results[group]}")
        json.dump(results, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/eval/toxigen")
    parser.add_argument("--save_dir", type=str, default="results/toxigen")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
    )
    parser.add_argument(
        "--classifier_batch_size",
        type=int,
        default=32,
        help="batch size to use for toxicity classifier.",
    )
    parser.add_argument(
        "--classifier_device",
        type=str,
        default="cuda",
        help="device to use for toxicity classifier.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, the prompt will be encoded as a chat format with the roles in prompt.",
    )
    parser.add_argument(
        "--use_llama_chat_format",
        action="store_true",
        help="If given, the prompt will be encoded as a chat format following llama 2.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
