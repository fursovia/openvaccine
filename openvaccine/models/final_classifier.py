from typing import Dict, Optional, Any

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, InputVariationalDropout, MatrixAttention
from allennlp.nn.util import get_text_field_mask, weighted_sum, masked_softmax
from allennlp.nn.regularizers import RegularizerApplicator

from openvaccine.losses import Loss
from openvaccine.modules import Aggregator


@Model.register("final_classifier")
class FinalClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            sequence_field_embedder: TextFieldEmbedder,
            structure_field_embedder: TextFieldEmbedder,
            predicted_loop_type_field_embedder: TextFieldEmbedder,
            seq2seq_encoder: Seq2SeqEncoder,
            loss: Loss,
            structure_field_attention: Optional[MatrixAttention] = None,
            predicted_loop_type_field_attention: Optional[MatrixAttention] = None,
            masked_lm: Optional[Model] = None,  # MaskedLanguageModel
            lm_is_trainable: bool = False,
            lm_matrix_attention: Optional[MatrixAttention] = None,
            lm_dropout: float = 0.0,
            emb_dropout: float = 0.0,
            bpps_aggegator: Optional[Aggregator] = None,
            bpp_dropout: float = 0.0,
            regularizer: RegularizerApplicator = None,
    ) -> None:
        super().__init__(vocab, regularizer)
        # embedders
        self._sequence_field_embedder = sequence_field_embedder
        self._structure_field_embedder = structure_field_embedder
        self._predicted_loop_type_field_embedder = predicted_loop_type_field_embedder

        self._seq2seq_encoder = seq2seq_encoder
        self._loss = loss

        self._structure_field_attention = structure_field_attention
        self._predicted_loop_type_field_attention = predicted_loop_type_field_attention

        # masked language models
        self._masked_lm = masked_lm
        if self._masked_lm is not None:
            self._masked_lm._tokens_masker = None
        if not lm_is_trainable and self._masked_lm is not None:
            self._masked_lm = self._masked_lm.eval()
        self._lm_matrix_attention = lm_matrix_attention

        self._lm_dropout = InputVariationalDropout(p=lm_dropout)
        self._emb_dropout = InputVariationalDropout(p=emb_dropout)

        self._bpps_aggegator = bpps_aggegator
        self._bpp_dropout = InputVariationalDropout(p=bpp_dropout)

        hidden_dim = self._seq2seq_encoder.get_output_dim()
        if self._masked_lm is not None:
            hidden_dim += self._masked_lm.get_output_dim()

        # we predict reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C
        self._linear = torch.nn.Linear(hidden_dim, 5)

    def forward(
            self,
            sequence: TextFieldTensors,
            structure: TextFieldTensors,
            predicted_loop_type: TextFieldTensors,
            seq_scored: torch.Tensor,
            seq_id: Any,
            bpps: Optional[torch.Tensor] = None,
            target: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sequence)
        sequence_embeddings = self._sequence_field_embedder(sequence)
        structure_embeddings = self._structure_field_embedder(structure)
        predicted_loop_type_embeddings = self._predicted_loop_type_field_embedder(predicted_loop_type)

        if self._structure_field_attention is not None:
            similarity_scores = self._structure_field_attention(
                sequence_embeddings,
                structure_embeddings
            )
            attention = masked_softmax(similarity_scores, mask, memory_efficient=False)
            structure_embeddings = weighted_sum(structure_embeddings, attention)

        if self._predicted_loop_type_field_attention is not None:
            similarity_scores = self._predicted_loop_type_field_attention(
                sequence_embeddings,
                predicted_loop_type_embeddings
            )
            attention = masked_softmax(similarity_scores, mask, memory_efficient=False)
            predicted_loop_type_embeddings = weighted_sum(predicted_loop_type_embeddings, attention)

        embeddings = torch.cat((sequence_embeddings, structure_embeddings, predicted_loop_type_embeddings), dim=-1)
        embeddings = self._emb_dropout(embeddings)

        if self._bpps_aggegator is not None:
            bpps = self._bpps_aggegator(bpps)
            bpps = self._bpp_dropout(bpps)
            embeddings = torch.cat((embeddings, bpps), dim=-1)

        contextual_embeddings = self._seq2seq_encoder(embeddings, mask)

        if self._masked_lm is not None:
            lm_contextual_embeddings = self._masked_lm(sequence, structure)["contextual_embeddings"]

            if self._lm_matrix_attention is not None:
                similarity_scores = self._lm_matrix_attention(contextual_embeddings, lm_contextual_embeddings)
                attention = masked_softmax(similarity_scores, mask, memory_efficient=False)
                att_vectors = weighted_sum(lm_contextual_embeddings, attention)

                contextual_embeddings = torch.cat(
                    [
                        contextual_embeddings,
                        att_vectors
                    ],
                    dim=-1
                )
            else:
                contextual_embeddings = torch.cat(
                    [
                        contextual_embeddings,
                        lm_contextual_embeddings
                    ],
                    dim=-1
                )

        logits = self._linear(contextual_embeddings)
        output_dict = dict(logits=logits, seq_id=seq_id)

        if target is not None:
            # TODO: slicing depends on whether we have start/end tokens
            # target.size(1) works since we know that all samples in the train
            # are of the same length
            target = target.transpose(1, 2)
            output_dict["loss"] = self._loss(
                logits[:, 1:target.size(1) + 1, :].reshape((-1, 5)),
                target.reshape((-1, 5)),
            )
        return output_dict
