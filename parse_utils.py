from dataclasses import dataclass
from simple_parsing import choice, field, ArgumentParser


@dataclass
class DataOptions:
    data_path: str = field("../data", alias="--data-path", help="Path to the dataset")
    results_path: str = field(
        "results", alias="--results-path", help="Path to store results"
    )
    data_name: str = choice(
        "celeba",
        "celebahq",
        "mnist",
        "cifar10",
        "lsun-church",
        "fashionmnist",
        default="cifar10",
        alias="--data-name",
        help="Name of the dataset",
    )


@dataclass
class TrainOptions:
    global_batch_size: int = field(
        128, alias="--global-batch-size", help="Global batch size"
    )
    iterations: int = field(400_000, alias="--iterations", help="Number of iterations")
    num_workers: int = field(
        4, alias="--num-workers", help="Number of workers for data loading"
    )
    log_every: int = field(500, alias="--log-every", help="Log every N iterations")
    ckpt_every: int = field(
        500, alias="--ckpt-every", help="Checkpoint every N iterations"
    )
    train_size: int = field(
        1_000, alias="--train-size", help="Size of the training set"
    )
    valid_size: int = field(
        1_000, alias="--valid-size", help="Size of the validation set"
    )
    global_seed: int = field(
        3407, alias="--global-seed", help="Global seed for reproducibility"
    )
    lr: float = field(1e-4, alias="--lr", help="Learning rate")
    ema_decay: float = field(0.9999, alias="--ema-decay", help="Decay value for EMA")
    clip_grad: bool = field(True, alias="--clip-grad", help="Whether to clip gradients")
    prediction_type: str = choice(
        "epsilon",
        "sample",
        default="epsilon",
        alias="--prediction-type",
        help="Prediction type for the score model",
    )
    beta_schedule: str = choice(
        "linear",
        default="linear",
        alias="--beta-schedule",
        help="Schedule for beta, can be linear, cosine, quadratic, or constant",
    )
    timesteps: int = field(1_000, alias="--timesteps", help="Number of diffusion steps")
    centercrop: bool = field(False, alias="--centercrop", help="Center crop the image")


@dataclass
class ModelOptions:
    image_size: int = field(32, alias="--image-size", help="Size of the image")
    in_channels: int = field(3, alias="--in-channels", help="Number of input channels")
    dim: int = field(128, alias="--dim", help="Initial Latent Dimension of the model")
    dim_mults: str = field(
        "1,2,2,2", alias="--dim-mults", help="Latent Dimension Multipliers"
    )
    attn_resolutions: str = field(
        "16", alias="--attn-resolutions", help="Attention Resolutions"
    )
    num_res_blocks: int = field(
        2, alias="--num-res-blocks", help="Number of Residual Blocks"
    )
    dropout: float = field(0.0, alias="--dropout", help="Dropout rate")
    conditional: bool = field(
        True, alias="--conditional", help="Whether to use conditional diffusion"
    )
    resamp_with_conv: bool = field(
        True, alias="--resamp-with-conv", help="Whether to use convolutional resampling"
    )
    nonlinearity: str = field(
        "swish", alias="--nonlinearity", help="Nonlinearity to use"
    )
    scale_by_sigma: bool = field(
        False, alias="--scale-by-sigma", help="Whether to scale by sigma"
    )
