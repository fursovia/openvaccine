from allennlp.common import Registrable
import torch


class Aggregator(torch.nn.Module, Registrable):

    def get_output_dim(self) -> int:
        pass

    def forward(self, bpps: torch.Tensor) -> torch.Tensor:
        pass


@Aggregator.register("max_mean_sum_agg")
class MaxMeanSumAggregator(Aggregator):

    def get_output_dim(self) -> int:
        return 3

    def forward(self, bpps: torch.Tensor) -> torch.Tensor:
        bpps = torch.cat(
            (
                bpps.max(dim=-1, keepdim=True).values,
                bpps.sum(dim=-1, keepdim=True),
                bpps.mean(dim=-1, keepdim=True),
            ),
            dim=-1
        )
        # start-end tokens
        bpps = torch.nn.functional.pad(bpps, [0, 0, 1, 1], value=0.0)
        return bpps
