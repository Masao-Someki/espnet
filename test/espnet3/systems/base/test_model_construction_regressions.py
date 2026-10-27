"""Training and statistics honor the public model construction configuration."""

from test.espnet3.components.modeling import test_model_with_optim_scheduler as helpers

import pytest
import torch
from omegaconf import OmegaConf

from espnet3.components.data.collect_stats import _build_model
from espnet3.systems.base.training import _build_trainer, _instantiate_model


class StrictModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)

    def collect_feats(self, x):
        return {"feats": x}


def test_statistics_strip_wrapper_options_without_mutating_config():
    config = OmegaConf.create(
        {
            "model": {
                "_target_": f"{__name__}.StrictModel",
                "freeze_param": ["encoder"],
            }
        }
    )
    assert isinstance(_instantiate_model(config), StrictModel)
    assert isinstance(
        _build_model(OmegaConf.create({"model_config": config.model})), StrictModel
    )
    assert config.model.freeze_param == ["encoder"]


@pytest.mark.parametrize("legacy", [False, True])
def test_top_level_initializer_reaches_real_trainer(tmp_path, monkeypatch, legacy):
    monkeypatch.setattr(
        helpers.data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: helpers.DummyDataset(),
    )
    config = helpers.make_single_config()
    config.exp_dir = str(tmp_path)
    config.best_model_criterion = [["valid/loss", 1, "min"]]
    config.trainer = helpers.make_trainer_config()
    config.model = {"_target_": f"{__name__}.StrictModel"}
    torch.manual_seed(7)
    baseline = _build_trainer(config).model.model.encoder.weight.detach().clone()
    if legacy:
        config.trainer.init = "xavier_uniform"
    else:
        config.init = "xavier_uniform"
    torch.manual_seed(7)
    initialized = _build_trainer(config).model.model.encoder.weight.detach()
    assert not torch.equal(baseline, initialized)
