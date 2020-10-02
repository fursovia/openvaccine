from overrides import overrides
import torch
from typing import List

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("stack")
class StackEncoder(Seq2SeqEncoder):

    def __init__(self, encoders: List[Seq2SeqEncoder]):
        super().__init__()
        self.encoders = encoders
        for idx, encoder in enumerate(encoders):
            self.add_module("encoder%d" % idx, encoder)

        # Compute bidirectionality.
        all_bidirectional = all(encoder.is_bidirectional() for encoder in encoders)
        any_bidirectional = any(encoder.is_bidirectional() for encoder in encoders)
        self.bidirectional = all_bidirectional

        if all_bidirectional != any_bidirectional:
            raise ValueError("All encoders need to match in bidirectionality.")

        if len(self.encoders) < 1:
            raise ValueError("Need at least one encoder.")

        input_dims = [enc.get_input_dim() for enc in self.encoders]
        assert len(set(input_dims)) == 1

        output_dims = [enc.get_output_dim() for enc in self.encoders]
        self.output_dim = sum(output_dims)

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        # Parameters

        inputs : `torch.Tensor`, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, timesteps).

        # Returns

        A tensor computed by composing the sequence of encoders.
        """
        outputs = []
        for encoder in self.encoders:
            outputs.append(encoder(inputs, mask))

        output = torch.cat(outputs, dim=-1)

        return output

    @overrides
    def get_input_dim(self) -> int:
        return self.encoders[0].get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional
