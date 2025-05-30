import os
import glob

import torch
import numpy as np
from tqdm.contrib import tzip
from train_utils import get_dataset
from simple_parsing import ArgumentParser

from pathlib import Path

from stats_utils import (
    sort_files,
    create_dirs,
    compute_metrics,
    get_train_loader,
    find_nns,
)

from typing import Tuple, List

"""Script for computing and storing memorized and spurious ratios along with the indices for nearest neighbors.
    Example usage:

        python run_classify.py --result-path images \
            --ratio-path ratios/ \ 
            --eval-path evals/ \
            --synth-path synths/ \
            --data-path data/ \
            --delta-m 0.3 \ 
            --delta-s 0.3 

    You could also use a config file instead and provide additional modifications to your arguments if needed.

    python run_metrics.py --config_path=my_config.py --data-path data/
"""


def get_top_least_idx(
    idx: np.array, top_n: int = 100, least_n: int = 100
) -> Tuple[np.array, Tuple[int, int]]:
    """
    Get the top and least indices from the given array.
    Args:
        idx: Numpy Array of integers with size N x k
        top_n: Number of top indices to extract
        least_n: Number of least indices to extract
    Returns:
        top_least_idx: Numpy Array of top and least indices
        sizes: Tuple of sizes of top and least indices
    """

    def extract(n, least: bool = False):
        n = min(len(idx), n)

        if least:
            least_idx = np.flip(idx, axis=0)[:n]
            return least_idx

        top_idx = idx[:n]
        return top_idx

    top_idx = extract(top_n, False)
    least_idx = extract(least_n, True)
    top_least_idx = np.concatenate([top_idx, least_idx], axis=0)
    return top_least_idx, (len(top_idx), len(least_idx))


def get_samples(
    idx: np.array,
    eval_set: np.array,
    synth_stats: Tuple[np.array, np.array],
    data_stats: Tuple[np.array, np.array],
) -> Tuple[np.array, np.array, np.array]:
    """
    Get samples from the eval set and their corresponding nearest neighbors from the synthetic and training sets.
    Args:
        idx: Numpy Array of integers with size N x k
        eval_set: Numpy Array of images in (-1, 1) scale
        synth_stats: Tuple of synthetic set and indices
        data_stats: Tuple of training set and indices
    Returns:
        samples: Numpy Array of images in (-1, 1) scale
        synth_nns: Numpy Array of nearest neighbors from the synthetic set
        data_nns: Numpy Array of nearest neighbors from the training set
    """
    data_set, data_indices = data_stats
    synth_set, synth_indices = synth_stats

    samples = find_nns(eval_set, idx)
    d_nn_idx = find_nns(data_indices, idx)
    s_nn_idx = find_nns(synth_indices, idx)

    d_nns = find_nns(data_set, d_nn_idx)
    s_nns = find_nns(synth_set, s_nn_idx)
    return samples, s_nns, d_nns


def save_samples(
    samples: np.array,
    train_nns: np.array,
    synth_nns: np.array,
    k: int,
    sizes: Tuple[int, int],
    path: str,
):
    """Save samples accordingly to a specified format
    Args:
        samples: Tensor consisting of images in (-1, 1) scale in shape of B x ...
        train_nns: Nearest neighbors from Training Set w.r.t samples in (-1, 1) scale of shape (B x K) x ...
        synth_nns: Nearest neighbors from Synthetic Set w.r.t samples in (-1, 1) scale of shape (B x K) x ...
    """
    result = {
        "samples": samples,
        "train-nns": train_nns,
        "synth-nns": synth_nns,
        "k": k,
        "sizes": sizes,  # (top size, least size)
    }
    np.savez_compressed(path, **result)


def filter_files(target: List[str], reference: List[str]) -> List[str]:
    """
    Filter files based on their sizes.
    Args:
        target: List of target files
        reference: List of reference files
    Returns:
        List of filtered files
    """

    def get_sizes(files):
        sizes = []
        for path in files:
            size = int(Path(path).name.split(".")[0])
            sizes.append(size)
        return sizes

    tar_sizes = get_sizes(target)
    ref_sizes = set(get_sizes(reference))

    files = []
    for i, size in enumerate(tar_sizes):
        if size in ref_sizes:
            files.append(target[i])
    return files


def main(args):
    result_path = args.result_path
    s_path = os.path.join(result_path, "spurious")
    m_path = os.path.join(result_path, "memorized")
    g_path = os.path.join(result_path, "generalized")
    create_dirs([s_path, m_path, g_path])

    if args.synth_path is None:
        args.synth_path = args.eval_path

    eval_files, synth_files, dist_files = map(
        lambda path: glob.glob(os.path.join(path, "*.npz")),
        (args.eval_path, args.synth_path, args.dist_path),
    )

    # sort files based on their prefix integer
    eval_files, synth_files, dist_files = map(
        sort_files, (eval_files, synth_files, dist_files)
    )

    # match files with their corresponding prefix
    eval_files, synth_files = map(
        lambda files: filter_files(files, dist_files), (eval_files, synth_files)
    )

    assert len(eval_files) == len(synth_files) and len(synth_files) == len(dist_files)

    start_idx = args.start_idx
    final_idx = args.final_idx
    if final_idx == -1:
        final_idx = len(dist_files)

    eval_files = eval_files[start_idx:final_idx]
    synth_files = synth_files[start_idx:final_idx]
    dist_files = dist_files[start_idx:final_idx]

    print(*zip(eval_files, synth_files, dist_files), sep="\n")

    dataset = None
    delta_s, delta_m = args.delta_s, args.delta_m
    for eval_path, synth_path, dist_path in tzip(eval_files, synth_files, dist_files):

        eval_ckpt = np.load(eval_path, allow_pickle=True)
        dist_ckpt = np.load(dist_path, allow_pickle=True)

        """"Load eval and synth sets"""
        eval_set = eval_ckpt["samples"]
        if eval_path != synth_path:
            synth_ckpt = np.load(synth_path, allow_pickle=True)
            synth_set = synth_ckpt["samples"]
        else:  # reuse memory for the case of eval and synth being the same
            synth_set = eval_set
            synth_ckpt = eval_ckpt
        ckpt_args = eval_ckpt["args"].item()

        assert (
            ckpt_args.train.train_size == synth_ckpt["args"].item().train.train_size
        ), "Train splits are not the same for the pair of files."

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

        # 1024 is the default batch size --- arbitrary value
        train_set = get_train_loader(dataset, 1024, ckpt_args).dataset
        data_size = len(train_set)

        mem_path, spu_path, gen_path = map(
            lambda z: os.path.join(z, str(data_size)), (m_path, s_path, g_path)
        )

        """Load distances
        m_dists: Tensor of shape (N x 1) with the top1-distances from the synthetic set to the training set
        s_dists: Tensor of shape (N x 1) with the top1-distances from the synthetic set to the synthetic set
        data_indices: Tensor of shape (N x k) with the indices of the training set
        synth_indices: Tensor of shape (N x k)
        """
        m_dists = torch.from_numpy(dist_ckpt["data-dists"][:, 0]).to("cuda")
        s_dists = torch.from_numpy(dist_ckpt["synth-dists"][:, 0]).to("cuda")
        data_indices = dist_ckpt["data-indices"][:, : args.k]
        synth_indices = dist_ckpt["synth-indices"][:, : args.k]

        """Compute Memorize, Spurious, and Generalize binary scores"""
        m_scores, s_scores, g_scores = compute_metrics(
            m_dists, s_dists, (delta_m, delta_s), None
        )

        """Sort scores based on their distances"""
        _, m_idx = torch.sort(m_dists, stable=True)
        _, s_idx = torch.sort(s_dists, stable=True)
        m_idx, s_idx = map(lambda z: z.cpu(), (m_idx, s_idx))

        """Save Memorized Samples"""
        k = data_indices.shape[1]
        if not os.path.exists(mem_path + ".npz") or args.overwrite:
            m_count = m_scores.sum().item()
            if m_count > 0:
                m_scores = m_scores[m_idx]
                m_idx = m_idx[torch.where(m_scores == 1)].numpy()
                m_idx, m_sizes = get_top_least_idx(
                    m_idx, args.top_size, args.least_size
                )

                m_samples, m_synth_nns, m_data_nns = get_samples(
                    m_idx,
                    eval_set,
                    (synth_set, synth_indices),
                    (train_set, data_indices),
                )

                save_samples(m_samples, m_data_nns, m_synth_nns, k, m_sizes, mem_path)

        if not os.path.exists(gen_path + ".npz") or args.overwrite:
            g_count = g_scores.sum().item()
            if g_count > 0:
                g_scores = g_scores[s_idx]
                g_idx = s_idx[torch.where(g_scores == 1)].numpy()

                # reverse the sorting of s_idx --- we want things far away from synthetic as most generalized
                g_idx = np.flip(g_idx, axis=0)

                g_idx, g_sizes = get_top_least_idx(
                    g_idx, args.top_size, args.least_size
                )

                g_samples, g_synth_nns, g_data_nns = get_samples(
                    g_idx,
                    eval_set,
                    (synth_set, synth_indices),
                    (train_set, data_indices),
                )

                save_samples(g_samples, g_data_nns, g_synth_nns, k, g_sizes, gen_path)

        if not os.path.exists(spu_path + ".npz") or args.overwrite:
            s_count = s_scores.sum().item()
            if s_count > 0:
                s_scores = s_scores[s_idx]
                s_idx = s_idx[torch.where(s_scores == 1)].numpy()
                s_idx, s_sizes = get_top_least_idx(
                    s_idx, args.top_size, args.least_size
                )

                s_samples, s_synth_nns, s_data_nns = get_samples(
                    s_idx,
                    eval_set,
                    (synth_set, synth_indices),
                    (train_set, data_indices),
                )

                save_samples(s_samples, s_data_nns, s_synth_nns, k, s_sizes, spu_path)


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_argument(
        "--result-path",
        type=str,
        help="Path to stored the results. If specified None, then results are stored in the same path as ckpt path.",
    )
    parser.add_argument(
        "--dist-path",
        type=str,
        help="Path to distance files (saved as .npz files)",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        help="Path to eval files (saved as .npz files)",
    )
    parser.add_argument(
        "--synth-path",
        type=str,
        help="Path to synthetic files (saved as .npz files)",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors.")
    parser.add_argument("--data-path", type=str, help="Path to the dataset folder.")
    parser.add_argument(
        "--delta-m", type=float, default=0.33, help="Memorization detection delta."
    )
    parser.add_argument(
        "--top-size", type=int, default=256, help="Top-k size to store."
    )
    parser.add_argument(
        "--least-size", type=int, default=64, help="Least-k size to store."
    )

    parser.add_argument(
        "--delta-s", type=float, default=0.33, help="Spurious detection delta."
    )
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
        "--overwrite", action="store_true", help="Overwrite existing files."
    )
    args = parser.parse_args()
    main(args)
