import logging
from typing import Tuple
from accelerate.logging import get_logger
import argparse
from easydict import EasyDict
import yaml
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from tqdm import tqdm
import torch
import torch.distributed as dist
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import datetime
import os
import transformers

# import my model
from modeling_my_qwen2 import Qwen2Config, Qwen2ForCausalLM

# setup cache and logger
torch._dynamo.config.cache_size_limit = 128

logger = get_logger(__name__)
logging.getLogger().handlers = []


# setup config file
def get_yaml_file(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file", default="train_lm_config.yaml")
    args = parser.parse_args()
    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    # Convert learning rate to float if it's a string
    if "lr" in yaml_config and isinstance(yaml_config["lr"], str):
        yaml_config["lr"] = float(yaml_config["lr"])

    yaml_config.update(args_dict)
    args = EasyDict(yaml_config)
    return args, yaml_config


# setup dataset
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, token_file: str, context_length: int, is_train: bool = True):
        """Initialize dataset with tokens array and context length"""
        self.x = np.load(token_file, mmap_mode="r")
        self.context_length = context_length
        self.is_train = is_train

    def __len__(self):
        """Return length of dataset"""
        if self.is_train:
            return len(self.x) - self.context_length
        else:
            return len(self.x) // self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example"""
        # Get sequence and target
        if self.is_train:
            idx = np.random.randint(0, len(self.x) - self.context_length)
        else:
            idx = idx * self.context_length
        x = torch.from_numpy(self.x[idx : idx + self.context_length].copy())
        y = x.clone()
        return x, y


def token_dataset_collate_fn(batch: list) -> dict:
    """Collate batch of examples"""
    # Stack sequences and targets
    x_batch = torch.stack([x for x, _ in batch]).long()
    y_batch = torch.stack([y for _, y in batch]).long()

    return {"input_ids": x_batch, "labels": y_batch}


def setup_dataloaders(args: EasyDict):
    train_dataset = TokenDataset(
        args.train_token_file, args.context_length, is_train=True
    )
    val_dataset = TokenDataset(args.val_token_file, args.context_length, is_train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=token_dataset_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=token_dataset_collate_fn,
    )
    return train_loader, val_loader


def setup_model(args: EasyDict):
    config = Qwen2Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        attention_dropout=args.attention_dropout,
    )
    model = Qwen2ForCausalLM(config)
    model.model = model.model.to(torch.bfloat16)
    return model


def get_optimizer(model: Qwen2ForCausalLM, args: EasyDict):
    optimizer_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            weight_decay = 0
        else:
            weight_decay = args.weight_decay
        optimizer_params.append(
            {"params": param, "lr": args.lr, "weight_decay": weight_decay}
        )
    optimizer = optim.AdamW(optimizer_params, betas=(0.9, 0.95), fused=True)
    return optimizer


# sched utils
def get_scheduler(optimizer, num_warmup_steps, num_training_steps, args: EasyDict):
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
        min_lr_multi=args.min_lr_multi,
    )
    return lr_scheduler


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_multi: float = 0.0,
    last_epoch: int = -1,
):
    """
    Modified from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        min_lr_multi (`float`, *optional*, defaults to 0):
            The minimum learning rate multiplier. Thus the minimum learning rate is base_lr * min_lr_multi.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(
                min_lr_multi, float(current_step) / float(max(1, num_warmup_steps))
            )
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_multi,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def validate_during_training(model, val_dataloader, accelerator):
    model.eval()
    total_loss = []
    for batch in tqdm(
        val_dataloader,
        total=len(val_dataloader),
        disable=not accelerator.is_local_main_process,
    ):
        outputs = model(**batch)
        loss = outputs["loss"]
        total_loss.append(loss.detach().float().cpu())
    model.train()
    if accelerator.use_distributed and accelerator.num_processes > 1:
        all_ranks_objects = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks_objects, total_loss)
        total_loss = [x for rank_loss in all_ranks_objects for x in rank_loss]
    val_loss = float(np.mean(total_loss))
    return val_loss


def main():
    args, args_dict = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_dir,
        kwargs_handlers=[kwargs],
    )

    if accelerator.is_local_main_process:
        log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        project_name = f"train_lm-{log_time}"
        accelerator.init_trackers(
            project_name=project_name,
            config=args_dict,
        )
        # Check if tensorboard tracker is actually available
        tracker = accelerator.get_tracker("tensorboard")
        LOG_DIR = os.path.join(args.output_dir, project_name)
        if accelerator.use_distributed and accelerator.num_processes > 1:
            print("rank", dist.get_rank(), f"TensorBoard log directory: {LOG_DIR}")
    else:
        LOG_DIR = ""

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    accelerator.wait_for_everyone()

    # debug
    if accelerator.is_local_main_process:
        print("args =", args)

    # get model
    model = setup_model(args)
    model.compile()
    params = model.num_parameters(only_trainable=True) / 1e6
    model = model.train()

    # get optimizer
    optimizer = get_optimizer(model, args)

    # get dataloaders
    train_loader, val_loader = setup_dataloaders(args)

    # prepare for accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    NUM_UPDATES_PER_EPOCH = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    MAX_TRAIN_STEPS = int(NUM_UPDATES_PER_EPOCH * args.max_epoch)
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    EVAL_STEPS = (
        args.val_interval
        if isinstance(args.val_interval, int)
        else int(args.val_interval * NUM_UPDATES_PER_EPOCH)
    )

    # get scheduler
    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=int(args.warmup_epochs * NUM_UPDATES_PER_EPOCH),
        num_training_steps=MAX_TRAIN_STEPS,
        args=args,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Model trainable parameters = {params}M")
    logger.info(f"  Num train examples = {len(train_loader) * args.train_batch_size}")
    logger.info(f"  Num dev examples = {len(val_loader) * args.val_batch_size}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.val_batch_size}")
    logger.info(f"  Eval steps = {EVAL_STEPS}")

    # training loop
    completed_steps = 0
    logging_interval_loss = {}
    total_loss = {}

    progress_bar = tqdm(
        range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process, ncols=100
    )

    save_metric_key = "val_loss"
    save_metric_value = None
    save_metric_larger_better = False

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed + epoch)
        progress_bar.set_description(f"epoch: {epoch + 1}")
        for step, batch in enumerate(train_loader):
            if completed_steps > MAX_TRAIN_STEPS:
                break
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                loss = outputs["loss"]
                logging_interval_loss.setdefault("loss", 0)
                logging_interval_loss["loss"] += loss.detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()

                    if args.logging_steps and completed_steps % args.logging_steps == 0:
                        avg_loss = {}
                        for k, v in logging_interval_loss.items():
                            avg_loss[k] = (
                                accelerator.gather(v).mean().item()
                                / args.gradient_accumulation_steps
                                / args.logging_steps
                            )
                        for k, v in logging_interval_loss.items():
                            total_loss.setdefault(k, 0)
                            total_loss[k] += (
                                accelerator.gather(v).mean().item()
                                / args.gradient_accumulation_steps
                            )

                        lr_string = f"{scheduler.get_last_lr()[-1]:.6f}"
                        loss_string = f"{avg_loss['loss']:.6f}"
                        progress_bar.set_postfix(loss=loss_string, lr=lr_string)

                        to_be_logged = {
                            "learning_rate": scheduler.get_last_lr()[-1],
                        }
                        for k, v in avg_loss.items():
                            to_be_logged[f"train_{k}"] = v
                        for k, v in total_loss.items():
                            to_be_logged[f"rolling_{k}"] = v / completed_steps
                        accelerator.log(to_be_logged, step=completed_steps)
                        logging_interval_loss = {}

                    if completed_steps % EVAL_STEPS == 0:
                        # evaluate
                        val_loss = validate_during_training(
                            model, val_loader, accelerator
                        )
                        val_results = {"val_loss": val_loss}
                        logger.info(f"validation loss = {val_loss}")
                        accelerator.log(val_results, step=completed_steps)

                        # save model
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                os.path.join(LOG_DIR, "model_last"),
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save,
                            )

                            if save_metric_key in val_results:
                                if save_metric_larger_better:
                                    if (
                                        save_metric_value is None
                                        or val_results[save_metric_key]
                                        > save_metric_value
                                    ):
                                        save_metric_value = val_results[save_metric_key]
                                        unwrapped_model.save_pretrained(
                                            os.path.join(LOG_DIR, "model_best"),
                                            is_main_process=accelerator.is_main_process,
                                            save_function=accelerator.save,
                                        )
                                else:
                                    if (
                                        save_metric_value is None
                                        or val_results[save_metric_key]
                                        < save_metric_value
                                    ):
                                        save_metric_value = val_results[save_metric_key]
                                        unwrapped_model.save_pretrained(
                                            os.path.join(LOG_DIR, "model_best"),
                                            is_main_process=accelerator.is_main_process,
                                            save_function=accelerator.save,
                                        )

    if accelerator.is_local_main_process:
        tracker.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
