"""Numerical optimizer updates, deferred windows and exact gradient resume."""

import copy
from test.espnet3.components.modeling import test_model_with_optim_scheduler as helpers

import pytest
import torch

from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.modeling.optimization_spec import OptimizationStep


class TraceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(2, 1, bias=False)
        self.discriminator = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(self.generator.weight)

    def forward(self, x):
        loss = self.generator(x).sum()
        return OptimizationStep(loss=loss, name="generator"), {}, None


@pytest.fixture(autouse=True)
def dataset_reference(monkeypatch):
    monkeypatch.setattr(
        helpers.data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: helpers.DummyDataset(),
    )


def _module(accum=1, every=1, reverse=False):
    config = helpers.make_multi_config(accum_grad_steps=accum, step_every_n_iters=every)
    for index, name in enumerate(["generator", "discriminator"]):
        config.optimizers[name].optimizer.lr = 1.0
        config.schedulers[name] = {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
                "gamma": [0.7, 0.2][index] if reverse else 1.0,
            },
            "interval": "step",
        }
    if reverse:
        config.schedulers = {
            name: config.schedulers[name] for name in ["discriminator", "generator"]
        }
    module = ESPnetLightningModule(TraceModel(), config)
    configured = module.configure_optimizers()
    helpers.prepare_manual_optimization(module, configured)
    return module, configured


def _batch(value):
    return (["utt"], {"x": torch.tensor([[value, value * 2]], dtype=torch.float32)})


def test_reversed_scheduler_order_uses_spec_names():
    module, configured = _module(reverse=True)
    named = module._get_named_schedulers()
    assert named["generator"] is configured[1][1]
    assert named["generator"].gamma == 0.7
    assert named["discriminator"].gamma == 0.2


@pytest.mark.parametrize("accum,every,count", [(1, 4, 4), (3, 2, 4), (2, 3, 3)])
def test_deferred_update_is_actual_window_mean(accum, every, count):
    module, _ = _module(accum, every)
    for index in range(count):
        module.training_step(_batch(index + 1), index)
    mean = (count + 1) / 2
    torch.testing.assert_close(
        module.model.generator.weight, torch.tensor([[-mean, -2 * mean]])
    )
    assert module._optimizer_states["generator"].accum_counter == 0


def test_epoch_end_flushes_partial_window():
    module, _ = _module(accum=2)
    for index, value in enumerate([1, 3, 5]):
        module.training_step(_batch(value), index)
    module.on_train_epoch_end()
    torch.testing.assert_close(
        module.model.generator.weight, torch.tensor([[-7.0, -14.0]])
    )
    assert module._optimizer_states["generator"].accum_counter == 0
    module.training_step(_batch(0), 0)
    torch.testing.assert_close(
        module.model.generator.weight, torch.tensor([[-7.0, -14.0]])
    )


def test_checkpoint_restores_pending_gradients_exactly():
    original, _ = _module(accum=3)
    original.training_step(_batch(1), 0)
    checkpoint = {}
    original.on_save_checkpoint(checkpoint)
    weights = copy.deepcopy(original.state_dict())
    resumed, _ = _module(accum=3)
    resumed.load_state_dict(weights)
    resumed.on_load_checkpoint(checkpoint)
    for index, value in enumerate([2, 3], start=1):
        original.training_step(_batch(value), index)
        resumed.training_step(_batch(value), index)
    torch.testing.assert_close(
        original.model.generator.weight, resumed.model.generator.weight
    )
    torch.testing.assert_close(
        resumed.model.generator.weight, torch.tensor([[-2.0, -4.0]])
    )


def test_legacy_checkpoint_discards_unrecoverable_partial_counter():
    module, _ = _module(accum=3)
    module.on_load_checkpoint(
        {
            "espnet3_optimizer_runtime_state": {
                "generator": {"accum_counter": 1, "update_step": 4},
            }
        }
    )
    assert module._optimizer_states["generator"].accum_counter == 0
    assert module._optimizer_states["generator"].update_step == 4
