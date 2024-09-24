import torch
import os
import pickle
import datasets

from typing import Any, Union, Optional
from dataclasses import dataclass

from datasets import load_dataset
from prompts import BASELINE_PROMPT, SINGLE_TOKEN_BASELINE_PROMPT
from torch.utils.data import Dataset, DataLoader

from peft import PeftConfig, PeftModel
from transformers.utils import PaddingStrategy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)


def get_dataset(tokenizer, train_task, exp_name):
    def generate_and_tokenize_prompt(data_point):
        if train_task == "raw":
            user_prompt = BASELINE_PROMPT.format(query=data_point["query"], apis=data_point["apis"])
            full_prompt = f"{user_prompt}[{data_point['pseudo_label']}]{tokenizer.eos_token}"
        elif train_task == "single_token":
            label_to_short = {"Answerable": "A", "Partially answerable": "P", "Unanswerable": "U"}
            user_prompt = SINGLE_TOKEN_BASELINE_PROMPT.format(query=data_point["query"], apis=data_point["apis"])
            full_prompt = f"{user_prompt}{label_to_short[data_point['pseudo_label']]}"
        else:
            print("train_task only support ['raw', 'single_token']")
            raise NotImplementedError

        tokenized_user_prompt = tokenizer(user_prompt, truncation=True, padding=True)
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) - 1

        tokenized_full_prompt = tokenizer(full_prompt, truncation=True, padding=True)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        tokenized_full_prompt["text"] = full_prompt

        return tokenized_full_prompt

    dataset_path = os.path.join(os.getcwd(), f"dataset_{train_task}_{exp_name}.pkl")
    if os.path.isfile(dataset_path):
        with open(dataset_path, "rb") as f:
            trainset, validset, plavset = pickle.load(f)
    else:
        dataset_name = "trainset.jsonl"
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.1)

        plav_eval = load_dataset("json", data_files="cgav_for_train_eval.jsonl", split="train")

        trainset = dataset["train"].shuffle().map(generate_and_tokenize_prompt)
        validset = dataset["test"].map(generate_and_tokenize_prompt)
        plav_eval = plav_eval.map(generate_and_tokenize_prompt)

        trainset = [d for d in trainset if len(d["input_ids"]) < 4096]
        validset = [d for d in validset if len(d["input_ids"]) < 4096]
        plav_eval = [d for d in plav_eval if len(d["input_ids"]) < 4096]

        filtered_trainset = {
            "input_ids": [item["input_ids"] for item in trainset],
            "attention_mask": [item["attention_mask"] for item in trainset],
            "labels": [item["labels"] for item in trainset],
        }
        filtered_validset = {
            "input_ids": [item["input_ids"] for item in validset],
            "attention_mask": [item["attention_mask"] for item in validset],
            "labels": [item["labels"] for item in validset],
        }
        filtered_plavset = {
            "input_ids": [item["input_ids"] for item in plav_eval],
            "attention_mask": [item["attention_mask"] for item in plav_eval],
            "labels": [item["labels"] for item in plav_eval],
        }

        trainset = datasets.Dataset.from_dict(filtered_trainset)
        validset = datasets.Dataset.from_dict(filtered_validset)
        plavset = datasets.Dataset.from_dict(filtered_plavset)

        with open(dataset_path, "wb") as f:
            pickle.dump([trainset, validset, plavset], f)
        print(f"Dataset saved at {dataset_path}")

    return trainset, validset, plavset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.labels = data["labels"]
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        return sample


def get_dataloader(dataset, pad_token_id, batch_size, shuffle):
    def left_pad_sequence(sequences, batch_first=False, padding_value=0):
        max_len = max(len(seq) for seq in sequences)
        padded_seqs = []
        for seq in sequences:
            padding = [padding_value] * (max_len - len(seq))
            padded_seq = padding + seq.tolist()  # 왼쪽에 패딩 추가
            padded_seqs.append(torch.tensor(padded_seq, dtype=seq.dtype))

        if batch_first:
            return torch.stack(padded_seqs)
        else:
            return torch.stack(padded_seqs).T  # (max_len, batch_size)

    # 패딩을 위한 collate_fn 정의
    def collate_fn(batch):
        labels = [item["labels"] for item in batch]
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        # Left padding
        padded_input_ids = left_pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        padded_attention_mask = left_pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_labels = left_pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }

    custom_dataset = CustomDataset(dataset)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader


@dataclass
class CustomDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        def left_pad_sequence(sequences, batch_first=False, padding_value=0):
            max_len = max(len(seq) for seq in sequences)
            padded_seqs = []
            for seq in sequences:
                padding = [padding_value] * (max_len - len(seq))
                padded_seq = padding + seq  # 왼쪽에 패딩 추가
                padded_seqs.append(torch.tensor(padded_seq, dtype=torch.long))

            if batch_first:
                return torch.stack(padded_seqs)
            else:
                return torch.stack(padded_seqs).T  # (max_len, batch_size)

        labels = [item["labels"] for item in features]
        input_ids = [item["input_ids"] for item in features]
        attention_mask = [item["attention_mask"] for item in features]

        # Left padding
        padded_input_ids = left_pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask = left_pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_labels = left_pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }


def get_model(model_path, device):
    lora_weights = model_path
    config = PeftConfig.from_pretrained(lora_weights)
    base_model_id = config.base_model_name_or_path

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        device_map=device,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model.eval()
