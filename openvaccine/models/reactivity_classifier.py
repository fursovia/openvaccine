from typing import Dict, Optional, Any, Tuple

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask

from openvaccine.reader import REACTIVITY_PADDING


def slice_logits(logits: torch.Tensor, reactivity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # logits: (bs, seq_len)
    # reactivity: (bs, scored_len)
    # scored_len <= seq_len

    not_padded = (reactivity != REACTIVITY_PADDING).float()
    scored_lengths = not_padded.sum(dim=1).long()

    logits = torch.cat(
        [
            l[1:scored_lengths[i] + 1].view(-1) for i, l in enumerate(logits)
        ]
    )

    reactivity = torch.cat(
        [
            r[:scored_lengths[i]].view(-1) for i, r in enumerate(reactivity)
        ]
    )

    return logits, reactivity


# TODO: rename to a single_classifier and allow to train on any single-sequence
@Model.register("reactivity_classifier")
class ReactivityClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            sequence_field_embedder: TextFieldEmbedder,
            structure_field_embedder: TextFieldEmbedder,
            seq2seq_encoder: Seq2SeqEncoder,
    ) -> None:
        super().__init__(vocab)
        self._sequence_field_embedder = sequence_field_embedder
        self._structure_field_embedder = structure_field_embedder
        self._seq2seq_encoder = seq2seq_encoder

        hidden_dim = self._seq2seq_encoder.get_output_dim()
        # we predict reactivity only
        self._linear = torch.nn.Linear(hidden_dim, 1)

    def forward(
            self,
            sequence: TextFieldTensors,
            structure: TextFieldTensors,
            seq_id: Optional[Any] = None,
            reactivity: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sequence)

        sequence_embeddings = self._sequence_field_embedder(sequence)
        structure_embeddings = self._structure_field_embedder(structure)

        embeddings = torch.cat((sequence_embeddings, structure_embeddings), dim=-1)

        contextual_embeddings = self._seq2seq_encoder(embeddings, mask)
        logits = self._linear(contextual_embeddings)

        output_dict = dict(
            logits=torch.cat(
                (
                    logits,
                    torch.zeros(logits.size(0), logits.size(1), 4, device=logits.device)
                ),
                dim=-1
            ),
            seq_id=seq_id
        )

        if reactivity is not None:
            logits, reactivity = slice_logits(logits, reactivity)

            output_dict["loss"] = torch.nn.functional.mse_loss(reactivity, logits)

        return output_dict
