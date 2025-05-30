import torch
from piqa import LPIPS
from typing import Callable
from piqa.utils.functional import l2_norm, reduce_tensor
from torchvision.transforms.functional import resize, InterpolationMode


class ModifiedLPIPS(LPIPS):
    """
    Example:
        module = ModifiedLPIPS()

        for _ in range(256):
            x, y = torch.randn(32, 3, 64, 64), torch.randn(256, 3, 64, 64)
            x.clamp_(0, 1)
            y.clamp_(0, 1)

            loss = module(x, y)
            print(loss.shape)
            print(loss.min(), loss.max())
    """

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        fn: Callable = None,
        unsqueeze: bool = False,
    ) -> torch.Tensor:

        if fn is not None:
            # assuming fn normalizes x and y back to (0, 1) range
            x, y = map(fn, (x, y))

        if x.size(1) == 1:
            # duplicate a single channel to 3 channels
            x, y = map(lambda z: z.repeat(1, 3, 1, 1), (x, y))

        if x.size(2) < 32:
            # if the image size is smaller than 32, resize to 32
            x, y = map(lambda z: resize(z, 32, InterpolationMode.NEAREST_EXACT), (x, y))

        # ImageNet normalization
        x, y = map(self.normalize, (x, y))

        # LPIPS
        lpips = 0.0

        for w, fx, fy in zip(self.weights, self.net(x), self.net(y)):
            fx = fx / (l2_norm(fx, dim=1, keepdim=True) + self.epsilon)
            fy = fy / (l2_norm(fy, dim=1, keepdim=True) + self.epsilon)

            if unsqueeze:
                fx = fx[:, None]
                fy = fy[None]

            # avg over H and W dims
            mse = (fx - fy).square().mean(dim=(-1, -2))
            score = mse @ w
            lpips += score
        return reduce_tensor(lpips, self.reduction)
