import os
import glob

import torch
import numpy as np
import torch.nn as nn

from tqdm.contrib import tzip
from train_utils import get_dataset
from diffusers import DDPMScheduler

from simple_parsing import ArgumentParser

from stats_utils import (
    sort_files,
    create_dirs,
    get_unet,
    get_metric_fn,
    get_train_loader,
)

from pathlib import Path
from einops import rearrange
from typing import Union, Callable, Optional


def select_sample(sample, size):
    """
    Selects a random sample of size `size` from the input sample.
    Args:
        sample: Input sample tensor.
        size: Size of the random sample to select.

    Returns:
    A tuple containing the selected sample and the indices used for selection.
    """
    idx = np.arange(len(sample))
    idx = np.random.choice(idx, size=size)
    return sample[idx], idx


@torch.no_grad()
def inject(scheduler, x, t):
    """Injects noise into the input tensor `x` at the specified timesteps `t` using the given scheduler.
    Args:
        scheduler: The scheduler to use for noise injection.
        x: Input tensor to inject noise into.
        t: Timesteps at which to inject noise.
    Returns:
        x_t: Tensor with injected noise.
    """
    noise = torch.randn_like(x)

    if type(t) is float:
        timesteps = torch.ones([x.shape[0]], device=x.device).long() * t
    else:
        timesteps = t
    x_t = scheduler.add_noise(x, noise=noise, timesteps=timesteps)
    return x_t


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


class DDIM:
    """
    DDIM Sampler for Denoising Diffusion Probabilistic Models (DDPM).
    This class implements the DDIM sampling algorithm, which is a non-Markovian
    variant of the DDPM. It allows for sampling from a trained diffusion model
    using the DDIM sampling strategy.
    """

    def __init__(self, diffusion: DDPMScheduler, model_fn: nn.Module):
        self.model_fn = model_fn
        self.diffusion = diffusion

    def T(self):
        return len(self.diffusion.timesteps)

    @torch.no_grad()
    def step(self, x, t, t_next, eta: float = 0.0):
        ndim = x.ndim

        # t, t_next = map(lambda z: z.to(x.device), (t, t_next))

        alphas_cumprod = self.diffusion.alphas_cumprod.to(x.device)
        at = append_dims(alphas_cumprod[t], ndim)

        at_next = torch.where(t_next >= 0, alphas_cumprod[t_next], 1.0)
        at_next = append_dims(at_next, ndim)

        # noise estimation
        et = self.model_fn(x, t)

        # predicts x_0 by direct substitution
        x0_t = (x - et * (1.0 - at).sqrt()) / at.sqrt()

        if eta > 0:
            # noise controlling the Markovia/Non-Markovian property
            sigma_t = eta * ((1.0 - at / at_next) * (1.0 - at_next) / (1.0 - at)).sqrt()
            perturbs = sigma_t * torch.randn_like(x)
        else:
            sigma_t = perturbs = 0.0

        # update using forward posterior q(x_t-1|x_t, x0_t)
        x = at_next.sqrt() * x0_t + (1.0 - at_next - sigma_t**2).sqrt() * et + perturbs
        return x, x0_t

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        denoise_steps: int,
        T: int = None,
        eta: float = 0,
        t0: int = 0,
    ):
        """Samples from the diffusion model using the DDIM sampling strategy.
        Args:
            x: Input tensor to sample from.
            t: Timesteps at which to sample.
            denoise_steps: Number of denoising steps to perform.
            T: Maximum diffusion time (optional).
            eta: Noise controlling the Markovia/Non-Markovian property (default: 0).
            t0: Minimum timestep (default: 0).
        Returns:
            x0: Sampled tensor after denoising.
        """

        # Assuming t has different timesteps
        if T is None:
            T = t.clone()

        denoise_steps = torch.where(t >= denoise_steps, denoise_steps, t)

        x0 = None
        t_now = t
        stride = torch.ceil(T / denoise_steps).long()
        stride = torch.where(stride == 0, 1, stride)

        while torch.all(t_now > t0):
            # clip t_now to avoid negative values
            t_now = torch.where(t_now < t0, t0, t_now)

            t_prev = t_now - stride
            x, x0 = self.step(x, t_now, t_prev, eta)
            t_now = t_prev

        return x0.clamp(-1, 1)


@torch.no_grad()
def find_opt_time(
    x: torch.Tensor,
    scheduler,
    sampler,
    denoise_steps: int,
    dist_metric: Callable,
    delta: float = 0.1,
    stride: int = 1,
    p_trials: int = 10,
    eta: float = 1.0,  # for DDIM
    p: float = 0.9,
    init_time: Optional[int] = None,
):
    """
    Find the optimal time for diffusion sampling based on the given parameters.
    Args:
        x: Input tensor to sample from.
        scheduler: Scheduler for the diffusion process.
        sampler: Sampler for the diffusion model.
        denoise_steps: Number of denoising steps to perform.
        dist_metric: Distance metric function to evaluate the error.
        delta: Threshold for stopping the sampling process (default: 0.1).
        stride: Step size for the sampling process (default: 1).
        p_trials: Number of trials for sampling (default: 10).
        eta: Noise controlling the Markovia/Non-Markovian property (default: 1.0).
        p: Probability threshold for stopping the sampling process (default: 0.9).
        init_time: Initial time for the sampling process (optional).
    Returns:
        Tc: Optimal time for diffusion sampling, which is the time in which you can still recover the perturbed sample within a threshold.
    """
    # timesteps
    t_start = 1

    # maximum diffusion time
    T = len(scheduler.timesteps)

    if init_time is not None:
        # init_time - stride, to avoid the scenario in which our init_time might already be Tc
        t_start = init_time - stride
        t_start = 1 if t_start <= 0 else t_start

    x0 = x.clone()
    Tc = torch.ones([len(x)]).long().to(x.device) * t_start
    not_stopped = torch.ones([len(x)]).to(x.device)

    for t in range(t_start, T, stride):
        t_now = torch.ones([len(x)]).to(x.device).long() * t

        # inject noise (forward process)
        noise = torch.randn_like(x)
        xt = scheduler.add_noise(x, noise=noise, timesteps=t_now)

        # reverse process applies at t_now
        pred = sampler.sample(xt, t_now, denoise_steps, eta=eta)

        # error between real sample x and recovered x_0 from t
        error = dist_metric(x0, pred)

        # if err is greater than delta, we stop
        recovered = torch.where(error <= delta, True, False)

        # probabilities of recovery across trials
        prob = rearrange(recovered, "(b n) -> b n", n=p_trials).float().mean(-1)

        # if recovered and we have not stopped, then continue else stopped by setting as 0
        # not_stopped = torch.where(torch.logical_and(prob > p, not_stopped == 1), 1, 0)
        not_stopped = torch.where(
            torch.logical_and(recovered == 1, not_stopped == 1), 1, 0
        )

        if torch.all(prob < p):
            break
        else:
            Tc = torch.where(not_stopped == True, t, Tc)
    return Tc


def main(args):
    result_path = args.result_path
    create_dirs([result_path])

    ckpt_files = glob.glob(os.path.join(args.ckpt_path, "*.pt"))

    if args.sample_path is not None:
        eval_files = glob.glob(os.path.join(args.sample_path, "*.npz"))

        # grab .pt or .pth files
        ckpt_files, eval_files = map(sort_files, (ckpt_files, eval_files))

        # filter out the appropriate files matching ckpt with eval
        new_files = []
        eval_sizes = set([int(Path(f).name.split(".")[0]) for f in eval_files])

        for f in ckpt_files:
            size = int(Path(f).name.split(".")[0])
            if size in eval_sizes:
                new_files.append(f)
        ckpt_files = new_files
        assert len(eval_files) == len(ckpt_files)
    else:
        eval_files = [None] * len(ckpt_files)
        ckpt_files = sort_files(ckpt_files)

    start_idx, final_idx = args.start_idx, args.final_idx
    if final_idx == -1:
        final_idx = len(ckpt_files)

    eval_files = eval_files[start_idx:final_idx]
    ckpt_files = ckpt_files[start_idx:final_idx]

    print(*list(zip(eval_files, ckpt_files)), sep="\n")

    dataset = diffusion = sampler = image_shape = None

    denoise_steps = args.ddim_steps
    metric_fn = get_metric_fn(args.use_lpips, args.network, unsqueeze=False)

    for ckpt_path, sample_path in tzip(ckpt_files, eval_files):
        model_ckpt = torch.load(ckpt_path, "cpu")
        ckpt_args = model_ckpt["args"]

        train_size = int(Path(ckpt_path).name.split(".")[0])
        save_path = os.path.join(result_path, str(train_size))

        if os.path.exists(save_path + ".npz") and not args.overwrite:
            continue

        if sample_path is not None:
            eval_ckpt = np.load(sample_path, allow_pickle=True)

        if image_shape is None:
            image_shape = [
                ckpt_args.model.in_channels,
                ckpt_args.model.image_size,
                ckpt_args.model.image_size,
            ]

            diffusion = DDPMScheduler(
                beta_schedule=ckpt_args.train.beta_schedule,
                prediction_type=ckpt_args.train.prediction_type,
            )
            diffusion.set_timesteps(ckpt_args.train.timesteps)

            dataset = get_dataset(
                args.data_path,
                ckpt_args.data.data_name,
                (
                    None
                    if not (ckpt_args.train.centercrop)
                    else ckpt_args.model.image_size
                ),
            )

        ema = get_unet(ckpt_args)
        ema.module.load_state_dict(model_ckpt["ema"])
        ema = ema.to("cuda")
        ema.eval()

        sampler = DDIM(diffusion, ema)

        if sample_path is not None:
            samples = eval_ckpt["samples"]
            top_size, least_size = eval_ckpt["sizes"]

            if args.use_least:
                samples = samples[-least_size:]
            else:
                samples = samples[:top_size]
            samples = torch.from_numpy(samples)

        else:
            train_loader = get_train_loader(dataset, args.sample_size, ckpt_args)
            for samples, _ in train_loader:
                break

        if len(samples) > args.sample_size:
            samples = samples[: args.sample_size]
        batch_size = min(args.batch_size, len(samples))

        times, radiuses = [], []
        for i in range(0, len(samples), batch_size):
            j = min(i + batch_size, len(samples))
            sample = samples[i:j].to("cuda")

            x0 = sample[:, None].repeat(1, args.p_trials, 1, 1, 1)
            x0 = rearrange(x0, "b n ... -> (b n) ...")

            opt_times = find_opt_time(
                x0,
                diffusion,
                sampler,
                denoise_steps,
                metric_fn,
                args.delta,
                args.stride,
                args.p_trials,
                args.eta,
                args.p,
            )

            x_t = inject(diffusion, x0, opt_times)  # (B x M) ...
            x_0, x_t = map(lambda z: torch.flatten(z, start_dim=1), (x0, x_t))

            radius = torch.norm(x_0 - x_t, dim=-1)

            radius, opt_times = map(
                lambda z: rearrange(z, "(b n) ... -> b n ...", n=args.p_trials),
                (radius, opt_times),
            )

            times.append(opt_times.cpu().numpy())
            radiuses.append(radius.cpu().numpy())

        times = np.concatenate(times, axis=0)
        radiuses = np.concatenate(radiuses, axis=0)
        np.savez_compressed(save_path, time=times, radius=radiuses, delta=args.delta)

        del ema
        torch.cuda.empty_cache()


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
        help="Path to evaluation set files (saved as .pt or .pth)",
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        default=None,
        help="Path to evaluation files (saved as .npz). If nothing is passed, evaluate training samples instead.",
    )
    parser.add_argument("--data-path", type=str, help="Path to the dataset folder.")
    parser.add_argument(
        "--sample-size", type=int, default=64, help="Size of the sample to evaluate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--use-lpips", action="store_true", help="Use LPIPS for distance metric."
    )
    parser.add_argument(
        "--use-least",
        action="store_true",
        help="Use bottom-k samples according to the distance.",
    )
    parser.add_argument(
        "--network", type=str, default="alex", help="Network to use for LPIPS."
    )
    parser.add_argument(
        "--p-trials", type=int, default=10, help="Number of trials for perturbation."
    )
    parser.add_argument(
        "--stride", type=int, default=25, help="Stride for time skipping."
    )
    parser.add_argument(
        "--ddim-steps", type=int, default=20, help="Number of DDIM steps."
    )
    parser.add_argument(
        "--delta", type=float, default=1e-1, help="Delta for stopping criterion."
    )
    parser.add_argument("--eta", type=float, default=0.0, help="Eta for DDIM.")
    parser.add_argument(
        "--p",
        type=float,
        default=0.9,
        help="Probability threshold for stopping criterion.",
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
        "--overwrite", action="store_true", help="Overwrite existing results."
    )
    args = parser.parse_args()
    main(args)
