from pathlib import Path

from allennlp.data import Vocabulary, Batch
from allennlp.common import Params
from allennlp.models import Model

import openvaccine


PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"


class TestCovidModel:

    def test_from_params(self):

        reader = openvaccine.CovidReader()
        instances = reader.read(PROJECT_ROOT / "data" / "sample.jsonl")
        vocab = Vocabulary.from_instances(instances)

        batch = Batch(instances)
        batch.index_instances(vocab)

        for config_path in CONFIG_DIR.glob("*.jsonnet"):
            try:
                params = Params.from_file(
                    str(config_path), ext_vars={"TRAIN_DATA_PATH": "", "VALID_DATA_PATH": ""}
                )
                model = Model.from_params(params=params["model"], vocab=vocab)
            except Exception as e:
                raise AssertionError(f"unable to load params from {config_path}, because {e}")

            output_dict = model(**batch.as_tensor_dict())

            assert set(output_dict.keys()) == {
                "logits",
                "seq_id",
                "loss",
            }

            assert len(output_dict["logits"].shape) == 3
            assert isinstance(output_dict["seq_id"][0], str)
