import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd

from loguru import logger
from collections import defaultdict
from prompts import *
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import precision_score, recall_score, f1_score


def parse_args(args):
    parser = argparse.ArgumentParser(description="Model evaluation script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, default="RaCGEval.json", help="Path to the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--token", type=str, help="Huggingface access token for gated models", default=None)
    parser.add_argument("--device", type=str, help="Device to use, e.g., cuda:0", default="cuda:0")
    parser.add_argument("--shots", type=str, choices=["zero", "one", "two"], help="Number of shots", default="zero")
    parser.add_argument("--print_every_response", action="store_true")
    parser.add_argument("--use_adapter", action="store_true")

    args = parser.parse_args(args)

    adpater_path = os.path.join(args.model_path, "adapter_config.json")
    if os.path.isfile(adpater_path):
        args.use_adapter = True

    return args


def set_custom_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def load_model(use_adapter, model_path, device):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if use_adapter:
        config = PeftConfig.from_pretrained(model_path)
        base_model_id = config.base_model_name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            config=config,
            device_map=device,
            attn_implementation="sdpa",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        model = PeftModel.from_pretrained(
            model,
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            padding_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            attn_implementation="sdpa",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
        )

    return tokenizer, model.eval()


def compute_metrics(labels, predictions):
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)

    return precision, recall, macro_f1, micro_f1


def get_prediction(model, tokenizer, final_prompt):
    device = model.device
    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
    logits = model(**inputs).logits.cpu().detach()

    labels = [" U", " P", " A"]
    tokens = tokenizer(labels)["input_ids"]

    # BOS token handling
    if tokens[0][0] == tokenizer.bos_token_id:
        tokens = list(map(lambda x: x[-1], tokens))
    logprobs = np.log(logits[0, -1].softmax(dim=-1))

    # Get the token which has maximum likelyhood
    token_probs = np.array([logprobs[token].item() for token in tokens])
    pred = np.argmax(token_probs)

    return pred


def log_and_save_results(datasets, label_dict, pred_dict, args):
    total_labels, total_preds, result_rows, lines = [], [], [], ""

    for dataset in datasets:
        labels = label_dict[dataset]
        predictions = pred_dict[dataset]

        precision, recall, macro_f1, micro_f1 = compute_metrics(labels, predictions)

        total_labels.extend(labels)
        total_preds.extend(predictions)

        line = f"{dataset}: Precision: {precision:.3f}, Recall: {recall:.3f}, Macro F1: {macro_f1:.3f}, Micro F1: {micro_f1:.3f}"
        logger.info(line)
        lines += line + "\n"
        result_rows.append([dataset, precision, recall, macro_f1, micro_f1])

    total_precision, total_recall, total_macro_f1, total_micro_f1 = compute_metrics(total_labels, total_preds)
    line = f"Total: Precision: {total_precision:.3f}, Recall: {total_recall:.3f}, Macro F1: {total_macro_f1:.3f}, Micro F1: {total_micro_f1:.3f}"
    logger.info(line)
    lines += line + "\n"
    result_rows.append(["Total", total_precision, total_recall, total_macro_f1, total_micro_f1])

    if args.use_adapter:
        save_path = os.path.join(args.model_path, f"racgeval_{args.shots}_shot.csv")
        line_save_path = os.path.join(args.model_path, f"racgeval_{args.shots}_shot.txt")
    else:
        save_dir = f"./result"
        os.makedirs(save_dir, exist_ok=True)
        model_name = args.model_path.split("/")[1]
        file_name = f"{model_name}_racgeval_{args.shots}_shot.csv"
        save_path = os.path.join(save_dir, file_name)

        txt_file_name = f"{model_name}_racgeval_{args.shots}_shot.txt"
        line_save_path = os.path.join(save_dir, txt_file_name)

    # Save results
    result_df = pd.DataFrame(result_rows, columns=["Dataset", "Precision", "Recall", "Macro F1", "Micro F1"])
    result_df.to_csv(save_path, index=False)
    with open(line_save_path, "w") as f:
        f.write(lines)
    logger.info(f"Results saved at {save_path}")


def main(args):
    tokenizer, model = load_model(args.use_adapter, args.model_path, args.device)

    with open(args.dataset_path, "r") as f:
        evalset = json.load(f)

    id_mapping = {
        "Unanswerable": 0,
        "Partially answerable": 1,
        "Answerable": 2,
    }

    label_dict, pred_dict = defaultdict(list), defaultdict(list)
    for data in tqdm(evalset.values()):
        query = data["query"]
        dataset = data["library"]
        label = data["label"]
        api_dict = data["retrieved_APIs"]
        apis = "\n".join([k + ": " + v for k, v in api_dict.items()])

        if args.shots == "zero":
            final_prompt = SINGLE_TOKEN_BASELINE_PROMPT.format(
                query=query,
                apis=apis,
            )
        else:
            if query in exception_query_list[args.shots]:
                continue
            final_prompt = prompt_mapping[dataset][args.shots].format(
                query=query,
                apis=apis,
            )

        pred = get_prediction(model, tokenizer, final_prompt)
        label_dict[dataset].append(id_mapping[label])
        pred_dict[dataset].append(pred)

    datasets = list(label_dict.keys())
    log_and_save_results(datasets, label_dict, pred_dict, args)


if __name__ == "__main__":
    args = parse_args(None)
    set_custom_seed(args.seed)

    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    main(args)
