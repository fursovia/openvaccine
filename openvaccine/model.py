from typing import Dict, Optional, Any

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, InputVariationalDropout, MatrixAttention
from allennlp.nn.util import get_text_field_mask, weighted_sum, masked_softmax

from openvaccine.losses import Loss


@Model.register("covid_classifier")
class CovidClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            sequence_field_embedder: TextFieldEmbedder,
            structure_field_embedder: TextFieldEmbedder,
            predicted_loop_type_field_embedder: TextFieldEmbedder,
            seq2seq_encoder: Seq2SeqEncoder,
            loss: Loss,
            variational_dropout: float = 0.0,
            matrix_attention: Optional[MatrixAttention] = None,
    ) -> None:
        super().__init__(vocab)
        self._sequence_field_embedder = sequence_field_embedder
        self._structure_field_embedder = structure_field_embedder
        self._predicted_loop_type_field_embedder = predicted_loop_type_field_embedder
        self._seq2seq_encoder = seq2seq_encoder

        self._loss = loss
        self._variational_dropout = variational_dropout
        self._dropout = InputVariationalDropout(p=self._variational_dropout)
        self._matrix_attention = matrix_attention

        hidden_dim = self._seq2seq_encoder.get_output_dim()
        if self._matrix_attention is not None:
            hidden_dim += self._sequence_field_embedder.get_output_dim() + \
                          self._structure_field_embedder.get_output_dim() + \
                          self._predicted_loop_type_field_embedder.get_output_dim()

        # we predict reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C
        self._linear = torch.nn.Linear(hidden_dim, 5)

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

        if self._variational_dropout > 0.0:
            embeddings = self._dropout(embeddings)

        contextual_embeddings = self._seq2seq_encoder(embeddings, mask)

        if self._matrix_attention is not None:
            similarity_scores = self._matrix_attention(contextual_embeddings, embeddings)
            attention = masked_softmax(similarity_scores, mask, memory_efficient=True)
            att_vectors = weighted_sum(embeddings, attention)

            contextual_embeddings = torch.cat(
                [
                    contextual_embeddings,
                    att_vectors
                ],
                dim=-1
            )

        logits = self._linear(contextual_embeddings)

        output_dict = dict(logits=logits, seq_id=seq_id)

        if target is not None:
            # TODO: slicing depends on whether we have start/end tokens
            target = target.transpose(1, 2)
            output_dict["loss"] = self._loss(
                logits[:, 1:target.size(1) + 1, :].reshape((-1, 5)),
                target.reshape((-1, 5)),
            )
        return output_dict
