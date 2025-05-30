import os
import torch
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src import Unet
from torch.nn import DataParallel
from train_utils import split_data

import torch.nn as nn

from pathlib import Path
from lpips import ModifiedLPIPS

from typing import List, Union, Callable, Tuple

"""
    Note: Assume samples are between (-1, 1)
"""


def get_unet(args, data_parallel: bool = True) -> nn.Module:
    """Create a UNet model with the given arguments
    Args:
        args: Command line arguments
        data_parallel: Whether to use DataParallel for the model
    Returns:
        model: A UNet model
    """
    model = Unet(
        image_size=args.model.image_size,
        in_channels=args.model.in_channels,
        dim=args.model.dim,
        dim_mults=args.model.dim_mults,
        attn_resolutions=args.model.attn_resolutions,
        num_res_blocks=args.model.num_res_blocks,
        dropout=args.model.dropout,
        conditional=args.model.conditional,
        resamp_with_conv=args.model.resamp_with_conv,
        nonlinearity=args.model.nonlinearity,
        scale_by_sigma=args.model.scale_by_sigma,
    )
    if data_parallel:
        model = DataParallel(model)
    return model


def create_dirs(paths: Union[str, List[str]]):
    """Create directories giving a list of paths
    Args:
        paths: a str or a list of strs indicating path(s) to create directories
    """
    if type(paths) == str:
        os.makedirs(paths, exist_ok=True)
    elif type(paths) == list:
        for path in paths:
            os.makedirs(path, exist_ok=True)
    else:
        raise NotImplementedError("Must be a str or a sequence of strs.")


def remove_pth(path: str) -> str:
    """Remove pth or pt extension from a string"""
    if path.endswith(".pth"):
        return path[:-4]
    elif path.endswith(".pt"):
        return path[:-3]
    else:
        raise Exception("Path does not end with .pth or .pt")


def sort_files(files: List[str]) -> List[str]:
    """Sort a bunch of strings using their suffix where we assume the suffix is a number
    E.g.,
    ./ckpts                 ./ckpts
        /unet_5.pt      =>      /unet_1.pt
        /unet_1.pt              /unet_5.pt
    """
    files.sort(key=lambda file: int(Path(file).name.split(".")[0]))
    return files


def to_zero_one(x: torch.Tensor) -> torch.Tensor:
    """Convert a (-1, 1) tensor to (0, 1) scale"""
    if x is None:
        return x
    x.mul_(0.5).add_(0.5)
    return x


def convert_dataset(train_loader: DataLoader) -> torch.Tensor:
    """Take a data iterator (with no shuffle enabled) and concat all of its samples"""
    xs = []
    for x, _ in train_loader:
        if len(x.shape) == 3:
            x = x[None]
        xs.append(x)
    return torch.cat(xs, dim=0)


def get_train_loader(
    dataset, batch_size: int, args, shuffle: bool = False
) -> DataLoader:
    """Given the Dataset object, return a dataloader of the training set of some given split size and batch size
    Args:
        dataset: Dataset object (e.g., MNIST, CIFAR10, etc.)
        batch_size: An integer determining the batch size
        args: Command line argument which includes a manual seed value and split sizes

    Notes: we set num_workers to 0, and pin_memory and drop_last to False
    """

    train_data, _ = split_data(
        dataset,
        args.train.train_size,
        args.train.valid_size,
        args.train.global_seed,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader


@torch.no_grad()
def compute_norm(
    x1: torch.Tensor, x2: torch.Tensor, unsqueeze: bool = True
) -> torch.Tensor:
    """Compute the l2 distance between two tensors
    Args:
        x1: Tensor of size b x ...
        x2: Tensor of size b x ...
        unsqueeze: Whether to add an extra dimension to the tensors
    Returns:
        Tensor of size b1 x b2
    """
    # convert from b x ... to b x (...)
    x1, x2 = map(lambda z: torch.flatten(z, start_dim=1), (x1, x2))

    if unsqueeze:
        x1 = x1[:, None]
        x2 = x2[None]

    # b1 x b2 x (...)
    diff = x1 - x2

    # b1 x b2
    return 1.0 / (torch.norm(diff, dim=-1) + 1e-8)


def get_metric_fn(
    use_lpips: bool,
    network: str = "alex",
    unsqueeze: bool = True,
    rescale: bool = True,
    device: str = "cuda",
) -> Callable:
    """Get the metric function for computing distances between two tensors
    Args:
        use_lpips: Whether to use LPIPS or not
        network: The network to use for LPIPS
        unsqueeze: Whether to add an extra dimension to the tensors
        rescale: Whether to rescale the tensors
        device: The device to use for computation
    Returns:
        metric_fn: A function which takes two tensors and computes their distance
    """
    if not rescale:
        rescale_fn = lambda z: z
    else:
        rescale_fn = lambda z: z * 0.5 + 0.5

    if use_lpips:
        metric = ModifiedLPIPS(network=network, reduction="none").to(device)
        metric_fn = lambda x, y: metric(x, y, fn=rescale_fn, unsqueeze=unsqueeze)
    else:
        metric_fn = lambda x, y: compute_norm(x, y, unsqueeze=unsqueeze)
    return metric_fn


@torch.no_grad()
def compute_metrics(
    m_top1_dists: torch.Tensor,
    s_top1_dists: torch.Tensor,
    deltas: Tuple[float, float],
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the memorized, spurious, and generalized binary scores

    Args:
        m_top1_dists: Tensor of size B
        s_top1_dists: Tensor of size B
        deltas: Tuple of two floats (delta_m, delta_s)
        device: Device to use for computation
    Returns:
        m_bins: Tensor of size B
        s_bins: Tensor of size B
        g_bins: Tensor of size B
    """
    delta_m, delta_s = deltas

    if device is not None:
        m_top1_dists, s_top1_dists = map(
            lambda z: z.to(device), (m_top1_dists, s_top1_dists)
        )

    # compute memorized scores
    m_bins = torch.where(m_top1_dists <= delta_m, 1, 0)
    # compute spurious scores
    s_bins = torch.where(s_top1_dists <= delta_s, 1, 0)

    neg_m_bins = m_bins ^ 1
    s_bins = torch.logical_and(neg_m_bins, s_bins)

    # compute generalized_scores
    g_bins = torch.logical_and(neg_m_bins, s_bins ^ 1)

    m_bins, s_bins, g_bins = map(lambda z: z.float().cpu(), (m_bins, s_bins, g_bins))
    return m_bins, s_bins, g_bins


def get_nonzero_entries(targets: torch.Tensor, binaries: torch.Tensor) -> torch.Tensor:
    """
    Given a binary vector of scores, find indices non-zero entries and grab the appropriate entries
    """
    indices = torch.where(binaries == 1)
    return targets[indices]


def find_nns(target, indices):
    """
    Given a list of indices, find the corresponding nearest neighbors in the target tensor
    Args:
        target: Tensor of size B x ...
        indices: List of indices
    Returns:
        nns: Tensor of size B x k x ...
    """
    if indices is None:
        return None

    elif len(indices) == 0:
        return None

    def not_tuple(item):
        if type(item) == tuple:
            return item[0]
        return item

    nns = []
    for idx in indices:
        try:
            nns.append(not_tuple(target[idx]))
        except:
            nn_i = []
            for i in idx:
                nn_i.append(not_tuple(target[i]))
            nn_i = np.stack(nn_i, axis=0)
            nns.append(nn_i)
    return np.stack(nns, axis=0)


@torch.no_grad()
def compute_rel_potential(
    model: nn.Module,
    target: torch.Tensor,
    reference: torch.Tensor,
    alpha_misc: Tuple[float, torch.Tensor, torch.Tensor],
    diff_misc: Tuple[torch.Tensor, torch.Tensor],
    mult_fn: nn.Module,
    t: int = 0,
):
    """
    Compute the relative potential between two tensors
    Args:
        model: Score function
        target: Target tensor of size B x ...
        reference: Reference tensor of size M x ...
        alpha_misc: Tuple of (dA, cos_alpha, sin_alpha)
        diff_misc: Tuple of (betas, std)
        mult_fn: A function to perform matrix multiplication
        t: Time step
    Returns:
        rel_potential: Tensor of size B x M
    """

    dA, cos_alpha, sin_alpha = alpha_misc
    betas, std = diff_misc

    b = len(target)
    p = len(cos_alpha)
    x = cos_alpha * target[None] + sin_alpha * reference[None]
    v = -sin_alpha * target[None] + cos_alpha * reference[None]

    x = rearrange(x, "p b ... -> (p b) ...")  # combine perturbations and batch
    t = torch.zeros(len(x), device=x.device).long() + t

    std_t = std[t]
    beta_t = betas[t][:, None, None, None]
    score = -model(x, t) / std_t[:, None, None, None]
    grad_u = -beta_t * score - 0.5 * beta_t * x
    grad_u = rearrange(grad_u, "(p b) ... -> p b ...", p=p, b=b)

    cumprod_u = 0

    v.mul_(dA)
    for i in range(p):
        val = mult_fn(
            grad_u[i].view(b, -1), v[i].view(b, -1)
        )  # return a vector of size b
        cumprod_u += val.sum(-1)

    rel_potential = cumprod_u
    return rel_potential.cpu().numpy()


def batch_potential(
    model: nn.Module,
    x1: torch.Tensor,  # assume this is a set of target images of size B
    x2: torch.Tensor,  # assume this is a set of reference images of size M
    betas: torch.Tensor,  # diffusion variances of size T,
    batch_size: int = 128,
    device: str = "cuda",
):
    """
    Given batches of images and a reference image,
    calculate the relative potential between those images and x_ref (batch wise),
    following (Spotaneous Symmetry ....) https://arxiv.org/pdf/2305.19693 paper,
    return all of the relative potentials of those images w.r.t to the reference image

    Args:
        model: Score function
        x1: Target images
        x2: A reference image
        betas: Diffusion betas
        batch_size: Number of images to evaluate at each iteration
    """

    class Model(nn.Module):
        """A simple torch Module object for DataParallel
        to perform matrix multiplication across multiple gpus
        """

        def __init__(self):
            super().__init__()

        def __call__(self, x1: torch.Tensor, x2: torch.Tensor):
            return x1 * x2

    assert batch_size > 0 and len(x2) == 1
    # according to paper https://arxiv.org/abs/2305.19693
    alpha = torch.linspace(0, 0.5 * torch.pi, 20)
    dA = (alpha[1] - alpha[0]).item()

    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)

    cos_alpha = cos_alpha[:, None, None, None, None].to(device)
    sin_alpha = sin_alpha[:, None, None, None, None].to(device)

    alpha_misc = (dA, cos_alpha, sin_alpha)

    rel_potentials = []
    model = model.eval()

    betas = betas.to(device)
    alpha_cps = torch.cumprod(betas, dim=0)

    diff_misc = (betas, (1 - alpha_cps) ** 0.5)
    mult_fn = nn.DataParallel(Model())

    # assume that x2 is smaller than x1 batch wise
    reference = x2.to(device)
    batch_size = min(len(x1), batch_size)

    for i in range(0, len(x1), batch_size):
        j = min(i + batch_size, len(x1))

        target = x1[i:j].to(device)
        rel_potential = compute_rel_potential(
            model, target, reference, alpha_misc, diff_misc, mult_fn
        )
        rel_potentials.append(rel_potential)

    if len(rel_potentials) == 1:
        return rel_potentials[0]
    return np.concatenate(rel_potentials)
