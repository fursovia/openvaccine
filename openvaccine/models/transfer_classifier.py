from typing import Dict, Optional, Any

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder

from openvaccine.losses import Loss


@Model.register("transfer_classifier")
class TransferCovidClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            masked_lm: Model,  # MaskedLanguageModel
            predicted_loop_type_field_embedder: TextFieldEmbedder,
            loss: Loss,
    ) -> None:
        super().__init__(vocab)
        self._masked_lm = masked_lm
        self._masked_lm._tokens_masker = None
        self._predicted_loop_type_field_embedder = predicted_loop_type_field_embedder

        self._loss = loss

        # we predict reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C
        self._linear = torch.nn.Linear(
            self._masked_lm.get_output_dim() + self._predicted_loop_type_field_embedder.get_output_dim(),
            5
        )

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
        predicted_loop_type_embeddings = self._predicted_loop_type_field_embedder(predicted_loop_type)

        contextual_embeddings = self._masked_lm(sequence, structure)["contextual_embeddings"]
        contextual_embeddings = torch.cat((contextual_embeddings, predicted_loop_type_embeddings), dim=-1)

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
