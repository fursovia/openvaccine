from typing import Optional

import torch
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.data import Vocabulary


@TokenEmbedder.register("onehot_embedder")
class OnehotEmbedder(TokenEmbedder):
    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        vocab: Optional[Vocabulary] = None,
        vocab_namespace: str = "sequence",
    ) -> None:
        super().__init__()
        assert embedding_dim is not None or vocab is not None
        self._embedding_dim = embedding_dim or vocab.get_vocab_size(vocab_namespace)

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = torch.nn.functional.one_hot(tokens, self._embedding_dim)
        return tokens.float()
