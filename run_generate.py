"""
The code is adapted from https://github.com/facebookresearch/DiT
for the generation of images using the diffusion model across multiple GPUs and nodes.
"""

import os
import glob

import string
import math
import torch
import numpy as np
from socket import gethostname

import torch.distributed as dist
from diffusers import DDPMScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from simple_parsing import ArgumentParser

from time import time
from stats_utils import get_unet

from src import *
from train_utils import sample, get_dist_info


def rand_str(chrs, size: int = 6):
    """
    Generate Random Characters
    Args:
        chrs (np.array): Array of characters to choose from.
        size (int): Number of characters to generate.
    Returns:
        str: Randomly generated string of characters.
    """
    return "".join(np.random.choice(chrs, size=size))


def main(rank, local_rank, args):
    world_size = dist.get_world_size()
    result_path = args.result_path

    diffusion = DDPMScheduler()  # default values as DDPM
    diffusion.set_timesteps(1000)  # 1000 sampling steps

    """Load model"""
    ckpt = torch.load(args.ckpt_path, map_location=f"cuda:{local_rank}")
    ckpt_args = ckpt["args"]

    """Reload model"""
    model = get_unet(ckpt_args, data_parallel=False)
    model = model.to(local_rank)
    ema = DDP(model, device_ids=[local_rank])
    ema.module.load_state_dict(ckpt["ema"])
    ema.eval()

    image_size = ckpt_args.model.image_size
    image_shape = [ckpt_args.model.in_channels, image_size, image_size]

    if rank == 0:
        print(f"Checkpoint: {args.ckpt_path}")
        os.makedirs(result_path, exist_ok=True)
        print(
            f"Saving samples in {result_path} and generating {args.num_files} .npz files.",
            flush=True,
        )
    dist.barrier()

    current_path = os.path.join(result_path, "*.npz")
    current_size = len(glob.glob(current_path))

    remaining = args.num_files - current_size

    iters = math.ceil(remaining / world_size)
    iters = 1 if iters == 0 else iters

    block_size = math.ceil(
        remaining / world_size
    )  # the number of images that each rank should generate

    idx = current_size + rank * block_size
    print(f"Rank: {rank}, Starting idx: {idx}", flush=True)

    chrs = [i for i in string.ascii_uppercase] + [i for i in string.ascii_lowercase]
    chrs = np.array(chrs)

    """Generation"""
    for i in range(iters):
        start_time = time()
        x = torch.randn(args.batch_size, *image_shape).to(local_rank)
        x = sample(diffusion, ema, x).cpu().numpy()

        if rank == 0:
            end_time = time()
            duration = end_time - start_time
            print(f"Iteration: {i}/{iters}, Time: {duration} sec", flush=True)

        rand_strs = rand_str(chrs, size=5)
        save_path = os.path.join(result_path, rand_strs)
        np.savez_compressed(save_path, samples=x)

    if rank == 0:
        print("Finished!", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_argument(
        "--result-path",
        type=str,
        help="Path to stored the results. If specified None, then results are stored in the same path as ckpt path.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="Path to a checkpoint file (saved as .pt or .pth)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size or the number of images per file.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument(
        "--num-files",
        type=int,
        default=10_000,
        help="Number of .npz files to generate.",
    )
    args = parser.parse_args()

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
