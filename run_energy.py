import os
import glob

import torch
import numpy as np

from tqdm.contrib import tzip
from train_utils import get_dataset
from diffusers import DDPMScheduler

from simple_parsing import ArgumentParser

from stats_utils import (
    sort_files,
    create_dirs,
    get_unet,
    get_train_loader,
    batch_potential,
)

from pathlib import Path


def main(args):
    result_path = args.result_path
    create_dirs([result_path])

    assert args.ref_path.endswith(".npz")
    references = torch.from_numpy(np.load(args.ref_path)["samples"])
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

    dataset = diffusion = image_shape = None
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

        if sample_path is not None:
            samples = eval_ckpt["samples"]
            top_size, least_size = eval_ckpt["sizes"]

            if args.use_least:
                # use bottom-k samples according to the distance
                samples = samples[-least_size:]
            else:
                # use top-k samples
                samples = samples[:top_size]
            samples = torch.from_numpy(samples)

        else:
            train_loader = get_train_loader(dataset, args.sample_size, ckpt_args)
            for samples, _ in train_loader:
                break

        if len(samples) > args.sample_size and not (args.sample_size < 0):
            samples = samples[: args.sample_size]
        batch_size = min(args.batch_size, len(samples))

        """Go over reference images and compute potentials."""
        potentials = []
        for ref in references:
            potential_i = batch_potential(
                ema, samples, ref[None], diffusion.betas, batch_size, "cuda"
            )
            potentials.append(potential_i)

        # store relative potentials per reference image
        potentials = np.vstack(potentials)
        np.savez_compressed(save_path, potential=potentials)

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
        "--ref-path",
        type=str,
        help="Path to a single .npz file containing reference images.",
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
        help="Path to evaluation files (saved as .npz). Sample set is obtained after run_classify.py is ran. If nothing is passed, evaluate training samples instead.",
    )
    parser.add_argument("--data-path", type=str, help="Path to the dataset folder.")
    parser.add_argument(
        "--sample-size", type=int, default=2048, help="Number of samples to evaluate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=384, help="Batch size to use for evaluation."
    )
    parser.add_argument(
        "--use-least",
        action="store_true",
        help="Use bottom-k samples according to the distance.",
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
