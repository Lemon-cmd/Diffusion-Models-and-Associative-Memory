"""
Fetch pre-trained models for the paper from the Hugging Face Hub.

Models are grouped into one repo per dataset and UNet width. Within a repo,
each file is named K.pt, where K is the size of the training set the model was
trained on. See https://huggingface.co/lemoncmd for the full collection.
"""

import os
import re
from typing import Sequence

HF_USER = "lemoncmd"
HF_PREFIX = "dm-am-"

# (data_name, dim) -> the group backing it on the Hub
GROUPS = {
    ("cifar10", 64): "cifar10-unet64",
    ("cifar10", 96): "cifar10-unet96",
    ("cifar10", 128): "cifar10-unet128",
    ("church64", 64): "church64-unet64",
    ("church64", 96): "church64-unet96",
    ("church64", 128): "church64-unet128",
    ("celebahq64", 64): "celebahq64-unet64",
    ("fmnist", 128): "fmnist-unet128",
    ("mnist", 128): "mnist-unet128",
}


def get_repo_id(data_name: str, dim: int) -> str:
    key = (data_name, int(dim))
    if key not in GROUPS:
        options = ", ".join(f"{d}/{k}" for d, k in sorted(GROUPS))
        raise KeyError(f"no released models for {data_name}/{dim}. Available: {options}")
    return f"{HF_USER}/{HF_PREFIX}{GROUPS[key]}"


def get_train_sizes(data_name: str, dim: int) -> Sequence[int]:
    """Every training-set size K released for this dataset and width."""
    from huggingface_hub import list_repo_files

    sizes = []
    for name in list_repo_files(get_repo_id(data_name, dim)):
        match = re.fullmatch(r"([0-9]+)\.pt", name)
        if match:
            sizes.append(int(match.groups()[0]))
    return sorted(sizes)


def download_checkpoint(data_name: str, dim: int, k: int, cache_dir: str = None) -> str:
    """Download the model trained on K samples. Returns the local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        get_repo_id(data_name, dim), f"{k}.pt", cache_dir=cache_dir
    )


def download_all(data_name: str, dim: int, local_dir: str = None) -> str:
    """Download every model for a dataset and width. Returns the local folder."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        get_repo_id(data_name, dim), allow_patterns=["*.pt"], local_dir=local_dir
    )


def strip_ddp_prefix(state_dict: dict) -> dict:
    """Checkpoints are saved from a DDP wrapper, so keys carry a module. prefix."""
    return {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}


def load_checkpoint(
    data_name: str, dim: int, k: int, ema: bool = True, map_location: str = "cpu"
) -> dict:
    """Download if needed, then return the weights for the model trained on K samples.

    ema=True returns the EMA weights, which are what the paper samples from.
    """
    import torch

    path = download_checkpoint(data_name, dim, k)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt["ema"] if ema else strip_ddp_prefix(ckpt["model"])


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", required=True, help="e.g. cifar10, church64, mnist")
    parser.add_argument("--dim", type=int, required=True, help="UNet width: 64, 96, or 128")
    parser.add_argument("--k", type=int, default=None, help="training size to fetch; omit for all")
    parser.add_argument("--result-path", default=None, help="where to place the files")
    parser.add_argument("--list", action="store_true", help="list available K, then exit")
    args = parser.parse_args()

    if args.list:
        sizes = get_train_sizes(args.data_name, args.dim)
        print(f"{get_repo_id(args.data_name, args.dim)}: {len(sizes)} models")
        print(" ".join(str(s) for s in sizes))
        return

    if args.k is not None:
        path = download_checkpoint(args.data_name, args.dim, args.k, cache_dir=args.result_path)
    else:
        path = download_all(args.data_name, args.dim, local_dir=args.result_path)
    print(path)


if __name__ == "__main__":
    main()
