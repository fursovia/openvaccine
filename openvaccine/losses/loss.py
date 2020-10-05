from allennlp.common import Registrable
import torch


class Loss(torch.nn.Module, Registrable):
    def __init__(self, calculate_on_scored: bool = True) -> None:
        super().__init__()
        self._num_to_calc_on = 3 if calculate_on_scored else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        pass
