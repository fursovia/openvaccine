import torch

from openvaccine.losses.loss import Loss


@Loss.register("MSE")
class MSE(Loss):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits[:, :self._num_to_calc_on]
        targets = targets[:, :self._num_to_calc_on]

        loss = torch.nn.functional.mse_loss(logits, targets)  # reduction="none")
        # loss = torch.mean(loss, dim=0).mean()
        return loss
