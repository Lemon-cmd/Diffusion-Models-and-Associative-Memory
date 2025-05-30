import os
import glob

import torch
import numpy as np

from tqdm import tqdm
from train_utils import get_dataset
from simple_parsing import ArgumentParser

from pathlib import Path

from stats_utils import (
    sort_files,
    create_dirs,
    get_train_loader,
    get_metric_fn,
)

from torch.utils.data import DataLoader

"""Script for computing and storing memorized and spurious ratios along with the indices for nearest neighbors.
    Example usage:

        python run_distances.py \
            --result-path distances \
            --synth-path cifar10-synths/ \
            --data-path data/ \
            --use-lpips 
            --k 5
            --network vgg

    You could also use a config file instead and provide additional modifications to your arguments if needed.
    python run_distances.py --config_path=my_config.py --data-path data/
"""


def dist_fn(sample, ref_set, metric_fn, batch_size, k, device="cuda"):
    """
    Compute Pair-wise Distances of elements in a given target set to a reference set
    Args:
        sample: Tensor of size B x ...
        ref_set:  Tensor of size M x ...
        metric_fn: a function which takes x, y for computing their distance to each other
        batch_size: the batch size for the reference set
        k: the number of nearest neighbors

    Returns:
        top_dist: B x k
        top_indices: B x k
    """
    same_set = True
    if type(ref_set) is not DataLoader:  # If given a synthetic set
        dists = torch.full([len(sample), len(ref_set)], float("inf"), device=device)

        for i in range(0, len(ref_set), batch_size):
            j = min(i + batch_size, len(ref_set))

            ref = torch.from_numpy(ref_set[i:j]).to(device)
            dists[:, i:j] = metric_fn(sample, ref)

    else:  # If given a training set
        i = 0
        same_set = False
        dists = torch.full(
            [len(sample), len(ref_set.dataset)], float("inf"), device=device
        )

        for ref, _ in ref_set:
            j = min(i + len(ref), len(ref_set.dataset))
            dists[:, i:j] = metric_fn(sample, ref.to(device))
            i = j

    k = min(
        dists.shape[1], k
    )  # for the case in which the number of samples is less than k

    if same_set:  # when we are computing synthetic to synthetic
        sk = min(k + 1, dists.shape[1])
        top_dists, top_idx = dists.topk(sk, dim=1, largest=False)
        return top_dists[:, 1:], top_idx[:, 1:]  # remove first column

    return dists.topk(k, dim=1, largest=False)


def get_top_dists(eval_set, ref_set, metric_fn, batch_sizes, k=1_000, device="cuda"):
    """
    Compute Pair-wise Distances of elements in a given target set to a reference set

    Args:
        eval_set: Tensor of size B x ...
        ref_set:  Tensor of size M x ...
        metric_fn: a function which takes x, y for computing their distance to each other
        k: the number of nearest neighbors

    Returns:
        top_dist: B x k
        top_indices: B x k
    """
    top_dists, top_indices = [], []
    eval_batch_size, ref_batch_size = batch_sizes
    ref_batch_size = eval_batch_size if ref_batch_size is None else ref_batch_size

    for i in range(0, len(eval_set), eval_batch_size):
        j = min(i + eval_batch_size, len(eval_set))
        target = torch.from_numpy(eval_set[i:j]).to(device)
        top_dist, top_index = dist_fn(
            target, ref_set, metric_fn, ref_batch_size, k, device
        )
        top_dists.append(top_dist)
        top_indices.append(top_index)

    top_dists, top_indices = map(torch.cat, (top_dists, top_indices))
    return top_dists.cpu(), top_indices.cpu()


def main(args):
    result_path = args.result_path
    create_dirs([result_path])

    # Grab .npz files for evaluation
    synth_files = glob.glob(os.path.join(args.synth_path, "*.npz"))
    synth_files = sort_files(
        synth_files
    )  # sort by name (or prefix converted into integer)

    # Slice the number of files for evaluation
    final_idx = len(synth_files) if args.final_idx == -1 else args.final_idx
    synth_files = synth_files[args.start_idx : final_idx]

    print(*synth_files, sep="\n")

    dataset = None
    metric_fn = get_metric_fn(args.use_lpips, network=args.network)

    lpips = "lpips" if args.use_lpips else "l2"
    backbone = args.network if args.use_lpips else "none"
    print(f"Using {lpips} with backbone: {backbone}")

    ref_batch_size = args.ref_batch_size
    eval_batch_size = args.eval_batch_size
    ref_batch_size = eval_batch_size if ref_batch_size is None else eval_batch_size

    # Go through each synthetic file and compute the distances
    for synth_path in tqdm(synth_files):
        synth_ckpt = np.load(synth_path, allow_pickle=True)
        ckpt_args = synth_ckpt["args"].item()

        # load the entire dataset a single time
        if dataset is None:
            dataset = get_dataset(
                args.data_path,
                ckpt_args.data.data_name,
                (
                    None
                    if not (ckpt_args.train.centercrop)
                    else ckpt_args.model.image_size
                ),
            )

        # split the dataset into a subset of K training samples
        train_loader = get_train_loader(dataset, ref_batch_size, ckpt_args)
        data_size = len(train_loader.dataset)
        save_path = os.path.join(result_path, f"{data_size}")

        if not (os.path.exists(save_path + ".npz")) or args.overwrite:
            synth_set = synth_ckpt["samples"]
            batch_sizes = (eval_batch_size, ref_batch_size)

            # compute synthetic to training set distances
            data_dists, data_indices = get_top_dists(
                synth_set, train_loader, metric_fn, batch_sizes, args.k
            )

            # compute synthetic to synthetic distances
            synth_dists, synth_indices = get_top_dists(
                synth_set, synth_set, metric_fn, batch_sizes, args.k
            )

            # save the results
            results = {
                "data-dists": data_dists.numpy(),  # synthetic to training set distances
                "synth-dists": synth_dists.numpy(),  # synthetic to synthetic distances
                "data-indices": data_indices.numpy(),
                "synth-indices": synth_indices.numpy(),
            }
            np.savez_compressed(save_path, **results)


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_argument(
        "--result-path",
        type=str,
        help="Path to stored the results. If specified None, then results are stored in the same path as ckpt path.",
    )
    parser.add_argument(
        "--synth-path", type=str, help="Path to evaluation files (saved as .npz)."
    )
    parser.add_argument("--data-path", type=str, help="Path to the dataset folder.")

    parser.add_argument(
        "--eval-batch-size", type=int, default=256, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--ref-batch-size",
        type=int,
        default=None,
        help="Batch size for reference set. If None, it is defaulted to eval_batch_size.",
    )

    parser.add_argument(
        "--use-lpips", action="store_true", help="Use LPIPS for distance."
    )
    parser.add_argument(
        "--network", type=str, default="alex", help="Backbone for LPIPS."
    )
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors.")
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting Index use for slicing the set of files we have to compute over.",
    )
    parser.add_argument(
        "--final-idx",
        type=int,
        default=-1,
        help="Ending Index use for slicing the set of files we have to compute over.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results."
    )
    args = parser.parse_args()
    main(args)
