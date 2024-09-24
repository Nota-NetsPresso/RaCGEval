import os
import random
import torch
import wandb
import datetime
import numpy as np

from peft import LoraConfig
from loguru import logger
from dataclasses import dataclass, field

from trl import SFTTrainer
from trl.commands.cli_utils import TrlParser

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    TrainingArguments,
)
from utils import get_dataset, CustomDataCollatorForSeq2Seq


def set_custom_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset"},
    )
    model_id: str = field(default=None, metadata={"help": "Model ID to use for SFT training"})
    max_seq_length: int = field(default=4096, metadata={"help": "The maximum sequence length for SFT Trainer"})
    train_task: str = field(default=None, metadata={"help": "Token to use(raw label or single token)"})


@dataclass
class LogArguments:
    exp_name: str = field(default=None, metadata={"help": "Path to the dataset"})
    output_base_dir: str = field(default=None, metadata={"help": "Base directory for model saving"})


def training_function(script_args, training_args, log_args):
    ################
    # TOKENIZER
    ################
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ################
    # DATASET
    ################
    train_dataset, test_dataset, plav_eval = get_dataset(tokenizer, script_args.train_task, log_args.exp_name)

    ################
    # MODEL
    ################
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset={"stblib": test_dataset, "plav": plav_eval},
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
            "skip_prepare_dataset": True,
        },
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((LogArguments, ScriptArguments, TrainingArguments))
    log_args, script_args, training_args = parser.parse_args_and_config()

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    current_datetime = datetime.datetime.now()
    hours_to_add = datetime.timedelta(hours=9)
    result_datetime = current_datetime + hours_to_add
    formatted_datetime = result_datetime.strftime("%m-%d_%H:%M:%S")

    exp_postfix = f"bs{training_args.per_device_train_batch_size}_acc{training_args.gradient_accumulation_steps}_gpu{world_size}"
    exp_name = f"{log_args.exp_name}_{script_args.train_task}_{exp_postfix}_input_{formatted_datetime}"
    output_dir = os.path.join(log_args.output_base_dir, exp_name)
    training_args.output_dir = output_dir

    if global_rank == 0:
        wandb.init(project="plav-vf-model-cls", name=exp_name)

        logger.info("log parameters")
        for k, v in vars(log_args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

        logger.info("training parameters")
        for k, v in vars(script_args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

        logger.info("training parameters")
        for k, v in vars(training_args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)
    set_custom_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args, log_args)
