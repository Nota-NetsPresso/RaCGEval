import os
import torch
import argparse
import numpy as np

from prompts import SINGLE_TOKEN_BASELINE_PROMPT
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="gemma-7b", choices=["gemma-7b", "llama3-8b"])
    parser.add_argument("--token", type=str, help="huggingface access token for gated models", default=None)
    parser.add_argument("--device", type=str, help="gpu to use", default="cuda:0")

    args = parser.parse_args(args)
    return args


def get_model(args):
    adapter_path = os.path.join('ckpt', f"{args.model_name}-adapter")
    config = PeftConfig.from_pretrained(adapter_path)
    base_model_id = config.base_model_name_or_path

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        device_map=args.device,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    # Load adapter & merge with base model
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model = model.merge_and_unload()
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model.eval()


def get_response(model, tokenizer, final_prompt):
    device = model.device
    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
    logits = model(**inputs).logits

    labels = [" U", " P", " A"]
    tokens = tokenizer(labels)["input_ids"]

    # BOS token handling
    tokens = list(map(lambda x: x[-1], tokens))

    # Find the token value with the maximum probability
    probs = np.array([logits[0, -1][token].item() for token in tokens])
    max_index = np.argmax(probs)

    # Answer mapping
    choices = ["[Unanswerable]", "[Partially answerable]", "[Answerable]"]
    return choices[max_index]


def main(args):
    tokenizer, model = get_model(args)

    EXAMPLE_INPUT = {
        "query": "How to augument the datapipe by repeating it six times.",
        "retrieved_APIs": {
            "API_1": "flatmap(*args, **kwds): Applies a function over each item from the source DataPipe, then flattens the outputs to a single, unnested IterDataPipe.",
            "API_2": "cycle(*args, **kwds): Cycles the specified input in perpetuity by default, or for the specified number of times.",
            "API_3": "mux(*datapipes): Yields one element at a time from each of the input Iterable DataPipes.\nfrom torchdata.datapipes.iter import IterableWrapper\ndatapipe = IterableWrapper([1,2,3])",
        },
    }

    api_string = "\n".join([f"{key}:{val}" for key, val in EXAMPLE_INPUT["retrieved_APIs"].items()])
    final_prompt = SINGLE_TOKEN_BASELINE_PROMPT.format(query=EXAMPLE_INPUT["query"], apis=api_string)
    response = get_response(model, tokenizer, final_prompt)

    print('APIs:')
    print(api_string, end='\n\n')
    print(f'Query: {EXAMPLE_INPUT['query']}')
    print(f'Response: {response}')


if __name__ == "__main__":
    args = parse_args(None)

    main(args)
