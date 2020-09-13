import torch

from openvaccine.losses.loss import Loss


@Loss.register("MCRMSE")
class MCRMSE(Loss):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits[:, :self._num_to_calc_on]
        targets = targets[:, :self._num_to_calc_on]

        mse = torch.nn.functional.mse_loss(logits, targets, reduction="none")
        mean_mse = torch.mean(mse, dim=0)
        sqrt_mse = torch.sqrt(mean_mse)
        loss = torch.mean(sqrt_mse)
        return loss
