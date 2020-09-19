from typing import List, Optional
import logging
import math
from pathlib import Path

import numpy as np
import jsonlines
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer, Token

logger = logging.getLogger(__name__)

START_TOKEN = "<START>"
END_TOKEN = "<END>"


@DatasetReader.register("covid_reader")
class CovidReader(DatasetReader):
    def __init__(
            self,
            bpps_dir: Optional[str] = None,
            max_sequence_length: Optional[int] = None,
            add_start_end_tokens: bool = True,
            lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)

        self._bpps_dir = bpps_dir if bpps_dir is None else Path(bpps_dir)
        self._max_sequence_length = max_sequence_length or math.inf
        self._add_start_end_tokens = add_start_end_tokens
        self._tokenizer = CharacterTokenizer()
        self._start_token = Token(START_TOKEN)
        self._end_token = Token(END_TOKEN)

    def _surround_with_start_end_tokens(self, tokens: List[Token]) -> List[Token]:
        return [self._start_token] + tokens + [self._end_token]

    def text_to_instance(
            self,
            sequence: str,
            structure: str,
            predicted_loop_type: Optional[str] = None,
            seq_scored: Optional[int] = None,
            seq_id: Optional[str] = None,
            reactivity: Optional[List[float]] = None,
            deg_Mg_pH10: Optional[List[float]] = None,
            deg_pH10: Optional[List[float]] = None,
            deg_Mg_50C: Optional[List[float]] = None,
            deg_50C: Optional[List[float]] = None,
            bpps: Optional[np.ndarray] = None,
            **kwargs
    ) -> Instance:

        sequence = self._tokenizer.tokenize(sequence)
        structure = self._tokenizer.tokenize(structure)

        if predicted_loop_type is not None:
            predicted_loop_type = self._tokenizer.tokenize(predicted_loop_type)

        if self._add_start_end_tokens:
            sequence = self._surround_with_start_end_tokens(sequence)
            structure = self._surround_with_start_end_tokens(structure)

            if predicted_loop_type is not None:
                predicted_loop_type = self._surround_with_start_end_tokens(predicted_loop_type)

        fields = {
            "sequence": TextField(sequence, {"tokens": SingleIdTokenIndexer("sequence")}),
            "structure": TextField(structure, {"tokens": SingleIdTokenIndexer("structure")}),
        }

        if predicted_loop_type is not None:
            fields["predicted_loop_type"] = TextField(
                predicted_loop_type,
                {"tokens": SingleIdTokenIndexer("predicted_loop_type")}
            )

        if seq_scored is not None:
            fields["seq_scored"] = LabelField(label=seq_scored, skip_indexing=True)

        if seq_id is not None:
            fields["seq_id"] = MetadataField(metadata=seq_id)

        if reactivity is not None:
            reactivity = np.array(reactivity)
            fields["reactivity"] = ArrayField(array=reactivity)

        if deg_Mg_pH10 is not None:
            deg_Mg_pH10 = np.array(deg_Mg_pH10)
            fields["deg_Mg_pH10"] = ArrayField(array=deg_Mg_pH10)

        if deg_pH10 is not None:
            deg_pH10 = np.array(deg_pH10)
            fields["deg_pH10"] = ArrayField(array=deg_pH10)

        if deg_Mg_50C is not None:
            deg_Mg_50C = np.array(deg_Mg_50C)
            fields["deg_Mg_50C"] = ArrayField(array=deg_Mg_50C)

        if deg_50C is not None:
            deg_50C = np.array(deg_50C)
            fields["deg_50C"] = ArrayField(array=deg_50C)

        if (
                reactivity is not None and
                deg_Mg_pH10 is not None and
                deg_pH10 is not None and
                deg_Mg_50C is not None and
                deg_50C is not None
        ):
            target = np.vstack((reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C))
            fields["target"] = ArrayField(array=target)

        if bpps is not None:
            fields["bpps"] = ArrayField(array=bpps)

        return Instance(fields)

    def _read(self, file_path: str):

        logger.info("Loading data from %s", file_path)
        dropped_instances = 0

        with jsonlines.open(cached_path(file_path), "r") as reader:
            for items in reader:
                sequence = items["sequence"]
                structure = items["structure"]
                assert len(sequence) == len(structure)

                predicted_loop_type = items.get("predicted_loop_type")
                if predicted_loop_type is not None:
                    assert len(predicted_loop_type) == len(structure)

                seq_id = items.get("seq_id")
                if self._bpps_dir is not None:
                    bpps = np.load(self._bpps_dir / f"{seq_id}.npy")
                else:
                    bpps = None

                instance = self.text_to_instance(
                    sequence=sequence,
                    structure=structure,
                    predicted_loop_type=predicted_loop_type,
                    seq_scored=items.get("seq_scored"),
                    seq_id=seq_id,
                    reactivity=items.get("reactivity"),
                    deg_Mg_pH10=items.get("deg_Mg_pH10"),
                    deg_pH10=items.get("deg_pH10"),
                    deg_Mg_50C=items.get("deg_Mg_50C"),
                    deg_50C=items.get("deg_50C"),
                    bpps=bpps,
                )
                if instance.fields["sequence"].sequence_length() <= self._max_sequence_length:
                    yield instance
                else:
                    dropped_instances += 1

        if not dropped_instances:
            logger.info(f"No instances dropped from {file_path}.")
        else:
            logger.warning(f"Dropped {dropped_instances} instances from {file_path}.")
