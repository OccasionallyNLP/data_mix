#!/usr/bin/env python
# coding=utf-8
"""
Pretrain language model
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import numpy as np
import datasets
import torch
import datetime

from dataloader import *

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import time

# XXX


logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=str,
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="If passed, will use a flash attention.",
    )
    
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--min_learning_rate_ratio",
        type=float,
        help="minimum learning rate_ratio",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs. Actual max train steps is multiply this value with num_processes. Total tokens of training is max_train_steps * num_proecesses * block_size * accumulation_steps * train_batch_size",
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=512,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup","cosine_with_min_lr"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument('--domain_weights_dict', type=str, required=True, help='Dictionary as a JSON string')
    parser.add_argument('--eval_domain_weights_dict', type=str, required=True, help='Dictionary as a JSON string')
    
    args = parser.parse_args()
    
    
    # JSON 문자열을 딕셔너리로 변환
    try:
        args.domain_weights_dict = json.loads(args.domain_weights_dict)
        args.eval_domain_weights_dict = json.loads(args.eval_domain_weights_dict)
        
    except json.JSONDecodeError as e:
        print("Invalid dictionary format. Please use a valid JSON string.")
        print("Error:", e)
        
    
    # Sanity checks
    if args.dataset_path is None :
        raise ValueError("Need either a dataset name or a training/validation file.")
    if args.preprocessing_num_workers is None:
        args.preprocessing_num_workers = int(os.cpu_count()*0.95)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    return args

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    
    ipg_handler = InitProcessGroupKwargs(
            timeout=datetime.timedelta(seconds=72000)
            )
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              kwargs_handlers=[ipg_handler], **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(args.output_dir,"plm.log")),  
    logging.StreamHandler()
    ])
    logger.info(accelerator.state, main_process_only=False)
    logger.info(str(args))
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name, trust_remote_code=args.trust_remote_code
            )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.flash_attention else None
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2" if args.flash_attention else None)

    #     We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    #     on a small vocab and want a smaller embedding size, remove this test.
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    ##########################################################################################################################
    # DataLoaders creation:
    # XXX
    train_dataset = get_preprocessed_mixed_dataset(
    args.dataset_path, 
    args.domain_weights_dict,
    cache_dir=args.cache_dir,
    split='train',
    seed=42,
    max_samples=None,
    add_domain_id=True,
    domain_weight_buffer_handle=None,
    tokenizer=tokenizer,
    no_interleave=False,
    shuffle=True,
    num_skip_examples=0,
    shard_reversal=False,
    keep_in_memory=False)
    
    print(train_dataset)
    
    # evaluation을 다르게 진행해야 할 듯?
    eval_dataset = get_preprocessed_mixed_dataset(
    args.dataset_path, 
    args.eval_domain_weights_dict,
    cache_dir=args.cache_dir,
    split='validation',
    seed=42,
    max_samples=None,
    add_domain_id=True,
    domain_weight_buffer_handle=None,
    tokenizer=tokenizer,
    no_interleave=True,
    shuffle=False,
    num_skip_examples=0,
    shard_reversal=False,
    keep_in_memory=False)
    
    print(eval_dataset)
    
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    ##########################################################################################################################
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # XXX
#     if args.max_train_steps is None:
#         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
#         overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes, # CHK
            num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes, # CHK
            scheduler_specific_kwargs={'min_lr_rate':args.min_learning_rate_ratio}
            if args.min_learning_rate_ratio
            else None
        )
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # XXX
#     num_update_steps_per_epoch = math.ceil()
    #math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
#     if overrode_max_train_steps:
#         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
#     args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config['domain_weights_dict'] = str(experiment_config['domain_weights_dict'])
        experiment_config['eval_domain_weights_dict'] = str(experiment_config['eval_domain_weights_dict'])
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Num of tokens in examples = {len(train_dataset)*args.block_size}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total tokens = {format(args.max_train_steps*args.block_size*total_batch_size,',')}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    trained_tokens = 0
    while True :
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            batch = {i:j for i,j in batch.items() if i!='domain_id'}
            trained_tokens+=((torch.prod(torch.tensor(batch['input_ids'].shape))).item()* accelerator.num_processes)
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                current_loss = np.round(loss.item(),2)
                logger.info(f"steps : {completed_steps} tokens: {format(trained_tokens,',')} lr: {lr_scheduler.get_last_lr()[0]} current loss: {current_loss}")
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        # evaluation
        model.eval()
        # TODO domain 별로 나눠서 진행.
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {i:j for i,j in batch.items() if i!='domain_id'}
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}_state_dict"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            ### save model
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

if __name__ == "__main__":
    main()
