import os
import torch
import logging
import numpy as np
from torchvision import transforms

from PIL import Image
from torchvision.datasets import CIFAR10, ImageFolder, MNIST, LSUN, FashionMNIST


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def str2tuple(v):
    """
    Convert a string to a tuple of integers.
    """
    try:
        return tuple([int(v)])
    except:
        return tuple([int(c) for c in v.split(",")])


def str2bool(v):
    """
    Convert a string to a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    return False


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def create_logger(logging_dir, index: int = 1):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log_{index}.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def get_transform(image_size=None, single=False):
    """
    Get the transform for the dataset.
    Args:
        image_size: Size of the image.
        single: If True, use single channel.
    Returns:
        transform: Transform for the dataset.
    """

    mean = MEAN if not single else MEAN[0]
    std = STD if not single else STD[0]

    if image_size is None:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True),
            ]
        )

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std, inplace=True),
        ]
    )
    return transform


def get_celebahq(data_path, img_size: int = None):
    """
    Get the CelebA-HQ dataset.
    Args:
        data_path: Path to the dataset.
        img_size: Size of the image.
    """

    def is_valid_file(path):
        # Check if the file is a numpy file
        if path.endswith(".npy") or path.endswith(".npz"):
            return path
        raise NotImplementedError("Invalid file.")

    def loader_fn(path):
        try:
            # Assuming that data is stored as C x H x W
            arr = np.load(path, mmap_mode="r").squeeze(0).transpose(1, 2, 0)
        except:
            arr = np.load(path, mmap_mode="r").transpose(1, 2, 0)
        return Image.fromarray(arr)

    data = ImageFolder(
        data_path,
        transform=get_transform(img_size),
        loader=loader_fn,
        is_valid_file=is_valid_file,
    )
    return data


def get_dataset(data_path: str, name: str = "cifar10", img_size: int = None):
    """
    Get the dataset.
    Args:
        data_path: Path to the dataset.
        name: Name of the dataset.
        img_size: Size of the image.
    Returns:
        Dataset: The dataset.
    """
    name = name.lower()

    if name == "cifar10":
        return CIFAR10(
            data_path, train=True, download=True, transform=get_transform(None)
        )
    elif name == "celeba" or name == "imagenet":
        return ImageFolder(data_path, transform=get_transform(img_size))
    elif name == "celebahq":
        return get_celebahq(data_path, img_size)
    elif name == "mnist":
        return MNIST(data_path, download=True, transform=get_transform(None, True))
    elif name == "fashionmnist":
        return FashionMNIST(
            data_path, download=True, transform=get_transform(None, True)
        )
    elif name == "lsun-church":
        return LSUN(
            data_path,
            classes=["church_outdoor_train"],
            transform=get_transform(img_size),
        )
    else:
        raise NotImplementedError("Invalid Dataset.")


def split_data(data, memorize_size, validate_size, seed):
    """
    Split the dataset into train and validation sets.
    Args:
        data: Dataset to be split.
        memorize_size: Size of the training set.
        validate_size: Size of the validation set.
        seed: Random seed for reproducibility.
    Returns:
        train_data: Training set.
        valid_data: Validation set.
    """
    max_size = len(data)
    generator = torch.Generator().manual_seed(seed)

    if memorize_size >= max_size:
        _, valid_data = torch.utils.data.random_split(
            data, [max_size - validate_size, validate_size], generator=generator
        )
        return data, valid_data

    elif validate_size >= max_size:
        train_data, _ = torch.utils.data.random_split(
            data, [memorize_size, max_size - memorize_size], generator=generator
        )
        return train_data, data

    train_data, valid_data, _ = torch.utils.data.random_split(
        data,
        [memorize_size, validate_size, max_size - (memorize_size + validate_size)],
        generator=generator,
    )
    return train_data, valid_data


# for multi-node
def get_dist_info():
    """
    Get the rank, world size and gpus per node.
    """
    # handle when these variables are not set
    if (
        "SLURM_PROCID" not in os.environ
        or "SLURM_NTASKS" not in os.environ
        or "SLURM_GPUS_ON_NODE" not in os.environ
    ):
        rank = 0
        world_size = 1
        gpus_per_node = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"
    else:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    return rank, world_size, gpus_per_node


@torch.no_grad()
def to_identity(x):
    return x


@torch.no_grad()
def to_real(x, vae):
    x = vae.decode(x / 0.18215).sample
    return x


def sample(scheduler, model, x):
    """
    Sample from the model.
    Args:
        scheduler: Scheduler to use for sampling.
        model: Model to use for sampling.
        x: Input tensor.
    Returns:
        x0: Sampled tensor.
    """
    model.eval()
    for t in scheduler.timesteps:
        x = scheduler.scale_model_input(x, t)
        with torch.no_grad():
            score = model(x, t.repeat(x.shape[0]).to(x.device))
        x = scheduler.step(score, t, x).prev_sample
    return x


def train_loss(scheduler, model, x, prediction_type: str = "epsilon"):
    """
    Compute the training loss.
    Args:
        scheduler: Scheduler to use for training.
        model: Model to use for training.
        x: Input tensor.
        prediction_type: Type of prediction to use.
    Returns:
        loss: Computed loss
    """
    noise = torch.randn_like(x)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, [x.shape[0]], device=x.device
    )
    x_t = scheduler.add_noise(x, noise=noise, timesteps=timesteps)
    score = model(x_t, timesteps)

    if prediction_type == "epsilon":
        return torch.square(noise - score).mean()
    elif prediction_type == "sample":
        return torch.square(x - score).mean()
    else:
        raise NotImplementedError("Invalid Prediction Type.")


def sample_data(loader):
    """
    Sample data from the loader infinitely.
    Args:
        loader: DataLoader to sample from.
    Yields:
        Sampled data from the loader.
    """
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)
            yield next(loader_iter)
