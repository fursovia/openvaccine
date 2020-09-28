from typing import Tuple, Optional

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder


def get_paddings(kernel_size: int) -> Tuple[int, ...]:
    mappings = {
        2: (0, 0, 1, 0),
        3: (0, 0, 1, 1),
        4: (0, 0, 2, 1),
        5: (0, 0, 2, 2),
        6: (0, 0, 3, 2),
        7: (0, 0, 3, 3),
    }
    return mappings[kernel_size]


@Seq2SeqEncoder.register("same_cnn_encoder")
class CNNEncoder(Seq2SeqEncoder):

    def __init__(self, input_dim: int, out_dim: int, kernel_size: int = 3, bidirectional: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size

        self.conv = torch.nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.out_dim,
            kernel_size=kernel_size
        )
        self.paddings = list(get_paddings(kernel_size))
        self.bidirectional = bidirectional

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.out_dim

    def is_bidirectional(self):
        # this is a hack since ComposeEncoder requires all encoders to match in bidirectionality
        return self.bidirectional

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        embeddings = torch.nn.functional.pad(embeddings, pad=self.paddings)
        embeddings = torch.transpose(embeddings, 1, 2)
        conv_embeddings = self.conv(embeddings)
        conv_embeddings = torch.transpose(conv_embeddings, 1, 2)

        if mask is not None:
            conv_embeddings = conv_embeddings * mask.unsqueeze(dim=-1)

        return conv_embeddings
