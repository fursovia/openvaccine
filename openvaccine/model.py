from typing import Dict, Optional, Any

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask


@Model.register("covid_classifier")
class CovidClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            sequence_field_embedder: TextFieldEmbedder,
            structure_field_embedder: TextFieldEmbedder,
            predicted_loop_type_field_embedder: TextFieldEmbedder,
            seq2seq_encoder: Seq2SeqEncoder,
    ) -> None:
        super().__init__(vocab)
        self._sequence_field_embedder = sequence_field_embedder
        self._structure_field_embedder = structure_field_embedder
        self._predicted_loop_type_field_embedder = predicted_loop_type_field_embedder
        self._seq2seq_encoder = seq2seq_encoder

        # we predict reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        self._linear = torch.nn.Linear(self._seq2seq_encoder.get_output_dim(), 5)
        self._loss = torch.nn.MSELoss(reduction="none")

    def _calculate_mcrmse(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch_size, seq_length, 5)
        logits = logits.reshape((-1, 5))
        targets = targets.reshape((-1, 5))

        mse = self._loss(logits, targets)
        mean_mse = torch.mean(mse, dim=0)
        sqrt_mse = torch.sqrt(mean_mse)
        loss = torch.mean(sqrt_mse)
        return loss

    def forward(
            self,
            sequence: TextFieldTensors,
            structure: TextFieldTensors,
            predicted_loop_type: TextFieldTensors,
            seq_scored: torch.Tensor,
            seq_id: Any,
            target: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sequence)

        sequence_embeddings = self._sequence_field_embedder(sequence)
        structure_embeddings = self._structure_field_embedder(structure)
        predicted_loop_type_embeddings = self._predicted_loop_type_field_embedder(predicted_loop_type)

        embeddings = torch.cat((sequence_embeddings, structure_embeddings, predicted_loop_type_embeddings), dim=-1)

        contextual_embeddings = self._seq2seq_encoder(embeddings, mask)

        logits = self._linear(contextual_embeddings)

        # output_dict = dict(contextual_embeddings=contextual_embeddings, logits=logits, mask=mask)
        output_dict = dict(logits=logits, seq_id=seq_id)

        if target is not None:
            # TODO: take `seq_scored` into account
            max_length = 68
            output_dict["loss"] = self._calculate_mcrmse(
                logits[:, 1:max_length + 1, :],
                target.transpose(1, 2),
            )
        return output_dict
