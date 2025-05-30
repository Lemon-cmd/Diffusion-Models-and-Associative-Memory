"""
The Code is taken from https://github.com/facebookresearch/DiT
and modified to work with the diffusers library.
This code is a PyTorch implementation of a training script for a diffusion model using the UNet architecture.
The script includes the following features:
- Distributed training using PyTorch's DDP (Distributed Data Parallel) module.
- Data loading and preprocessing using torchvision.
- Logging and checkpointing of the model's state during training.
- Sampling from the model at regular intervals to generate images.
- Support for different beta schedules and prediction types for the diffusion process.
- Command-line argument parsing using simple_parsing.
"""

import os
import torch
import glob
import numpy as np
from time import time
from socket import gethostname

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src import *

from train_utils import (
    get_dataset,
    str2tuple,
    create_logger,
    train_loss,
    sample,
    split_data,
    sample_data,
    get_dist_info,
)

from stats_utils import get_unet

from parse_utils import TrainOptions, DataOptions, ModelOptions

from collections import OrderedDict
from diffusers import DDPMScheduler
from simple_parsing import ArgumentParser

from copy import deepcopy


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad == True:
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def main(rank, local_rank, args):
    assert (
        args.train.global_batch_size % dist.get_world_size() == 0
    ), "Batch size must be divisible by world size."

    experiment_dir = f"{args.data.results_path}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    sample_dir = f"{experiment_dir}/samples"
    logger_dir = f"{experiment_dir}/logs"

    if rank == 0:
        """Create directories for experiment, checkpoints, samples, and logs"""
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(logger_dir, exist_ok=True)

        logger_index = len(glob.glob(f"{logger_dir}/*"))
        logger = create_logger(logger_dir, logger_index + 1)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(
            f"Batch size per rank: {args.train.global_batch_size // dist.get_world_size()}"
        )

    channels = args.model.in_channels
    seed = args.train.global_seed * dist.get_world_size() + local_rank
    torch.manual_seed(seed)

    """Create model and optimizer"""
    model = get_unet(args, data_parallel=False)

    # create EMA model
    ema = deepcopy(model).to(local_rank)
    requires_grad(ema, False)
    ema.eval()

    # set to local rank for DDP
    model = model.to(local_rank)
    update_ema(ema, model, decay=0)  # initialize EMA model
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.Adam(model.parameters(), lr=args.train.lr)

    iterations = 0
    latest_checkpoint = get_latest_file(checkpoint_dir, ".pt")
    if latest_checkpoint is not None:
        """Load latest checkpoint"""
        if rank == 0:
            logger.info(f"Found latest checkpoint file: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint, map_location=f"cuda:{local_rank}")
        iterations = checkpoint["iterations"]
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        model.module.load_state_dict(checkpoint["model"])  # model is a DDP wrap around

    """Create diffusion scheduler"""
    diffusion = DDPMScheduler(
        beta_schedule=args.train.beta_schedule,
        prediction_type=args.train.prediction_type,
    )
    diffusion.set_timesteps(args.train.timesteps)

    """Create Data Loader"""
    dataset = get_dataset(
        args.data.data_path,
        args.data.data_name,
        None if not (args.train.centercrop) else args.model.image_size,
    )
    train_data, _ = split_data(
        dataset, args.train.train_size, args.train.train_size, args.train.global_seed
    )

    """Create Data Loader"""
    # shuffle must be set as True for this object but not DataLoader according to torch documentation
    sampler = DistributedSampler(
        train_data,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.train.global_seed + iterations,
    )

    batch_size = args.train.global_batch_size
    batch_size = min(batch_size, args.train.train_size)

    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        """Log memory usage and model summary"""
        logger.info(torch.cuda.memory_summary(abbreviated=True))
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(
            f"Training for {args.train.iterations} iterations with {args.train.train_size} samples."
        )
        logger.info(f"Batch size per rank: {batch_size}")

    loader = sample_data(loader)
    log_steps = running_loss = 0

    model.train()
    start_time = time()

    # set epoch for sampler for distributed training randomness
    sampler.set_epoch(iterations)
    for x, _ in loader:
        if iterations == args.train.iterations:
            break

        x = x.to(local_rank)
        loss = train_loss(diffusion, model, x, args.train.prediction_type)
        loss.backward()

        if args.train.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        opt.zero_grad()
        update_ema(ema, model.module, decay=args.train.ema_decay)

        log_steps += 1
        iterations += 1
        running_loss += loss.item()

        if iterations % 100 == 0:
            sampler.set_epoch(iterations)

        if (
            iterations % args.train.ckpt_every == 0
            or iterations == args.train.iterations
        ):
            if rank == 0:
                checkpoint_path = f"{checkpoint_dir}/{iterations}.pt"
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "iterations": iterations,
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

            if rank == 0:
                image_size = args.model.image_size
                x = torch.randn(
                    144, channels, image_size, image_size, device=local_rank
                )
                x = sample(diffusion, ema, x)

                sample_path = f"{sample_dir}/{iterations}.png"
                x = transforms.Resize((48, 48), antialias=False)(x)
                save_image(
                    x,
                    sample_path,
                    nrow=12,
                    normalize=True,
                    scale_each=True,
                    value_range=(-1, 1),
                )
                model.train()

        if iterations % args.train.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            sampler.set_epoch(iterations)
            steps_per_sec = end_time - start_time
            avg_loss = torch.tensor(running_loss / log_steps, device=local_rank)
            avg_loss = avg_loss.item() / dist.get_world_size()

            if rank == 0:
                logger.info(
                    f"(Step={iterations:07d}), Train Loss: {avg_loss:.5f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
            log_steps = running_loss = 0
            start_time = time()

    if rank == 0:
        logger.info("Finished.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(DataOptions, dest="data")
    parser.add_arguments(TrainOptions, dest="train")
    parser.add_arguments(ModelOptions, dest="model")
    args = parser.parse_args()

    args.model.dim_mults = str2tuple(args.model.dim_mults)
    args.model.attn_resolutions = str2tuple(args.model.attn_resolutions)

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    rank, world_size, gpus_per_node = get_dist_info()
    assert gpus_per_node == torch.cuda.device_count()

    print(
        f"Hello from Rank {rank} of {world_size} on {gethostname()} where there are"
        f" {gpus_per_node} allocated GPUs per node.",
        flush=True,
    )

    # gpu rank
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    # always do before init process group
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
    main(rank, local_rank, args)
