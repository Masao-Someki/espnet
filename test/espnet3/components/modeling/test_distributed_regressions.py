"""Two CPU ranks agree on data length, skip flags and sample-weighted gradients."""

import os
from datetime import timedelta

import lightning
import pytest
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from espnet3.components.data.dataloader import DataLoaderBuilder
from espnet3.components.data.epoch_sync_iterator import EpochSyncIterator
from espnet3.components.modeling.lightning_module import ESPnetLightningModule


class _Mean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        loss = (self.weight * x).mean()
        return loss, {}, torch.tensor(len(x))


class _Loss(ESPnetLightningModule):
    def __init__(self):
        lightning.LightningModule.__init__(self)
        self.model = _Mean()
        self.config = OmegaConf.create({"num_device": 1})
        self.nan_countdown = 0

    def _log_stats(self, *args, **kwargs):
        pass

    def forward(self, x):
        return self._step(([], {"x": x}), 0, "train")


class _Sized(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight * batch.mean()

    def validation_step(self, batch, batch_idx):
        self.log("valid/loss", self.weight, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _worker(rank, root, queue):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        GROUP_RANK="0",
        GROUP_WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29679",
    )
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{root}/group",
        timeout=timedelta(seconds=30),
    )
    try:
        module = _Loss()
        loader_builder = DataLoaderBuilder(None, OmegaConf.create({}), None, 1, 0)
        assert loader_builder._get_world_info() == (rank, 2)
        assert module._sync2skip(torch.tensor(rank == 0))
        model = torch.nn.parallel.DistributedDataParallel(module)
        values = torch.tensor([1.0, 3.0]) if rank == 0 else torch.tensor([5.0])
        model(values).backward()
        torch.testing.assert_close(module.model.weight.grad, torch.tensor(3.0))
        iterator = EpochSyncIterator([torch.ones(1)] * (2 + rank))
        assert len(iterator) == 2
        trainer = lightning.Trainer(
            accelerator="cpu",
            devices=2,
            strategy="ddp",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            default_root_dir=root,
        )
        trainer.fit(
            _Sized(),
            train_dataloaders=iterator,
            val_dataloaders=torch.utils.data.DataLoader([torch.ones(1)]),
        )
        queue.put((rank, trainer.global_step))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.execution_timeout(90)
def test_two_rank_training_contracts(tmp_path):
    queue = mp.get_context("fork").Queue()
    mp.start_processes(
        _worker, args=(str(tmp_path), queue), nprocs=2, start_method="fork", join=True
    )
    assert sorted(queue.get(timeout=5) for _ in range(2)) == [(0, 2), (1, 2)]
    queue.close()


def _manual_worker(rank, root, queue):
    from test.espnet3.components.modeling import (
        test_model_with_optim_scheduler as helpers,
    )

    from espnet2.train.collate_fn import CommonCollateFn
    from espnet3.components.modeling.optimization_spec import OptimizationStep

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        GROUP_RANK="0",
        GROUP_WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29680",
    )
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{root}/manual-group",
        timeout=timedelta(seconds=20),
    )
    try:

        class WeightedGAN(helpers.DummyGANModel):
            def forward(self, x, **kwargs):
                steps, stats, _ = super().forward(x, **kwargs)
                return (
                    [
                        OptimizationStep(loss=step.loss / len(x), name=step.name)
                        for step in steps
                    ],
                    stats,
                    torch.tensor(len(x)),
                )

        helpers.data_organizer_module.instantiate_dataset_reference = (
            lambda config, recipe_dir=None: helpers.DummyDataset()
        )
        config = helpers.make_multi_config(accum_grad_steps=2)
        for name in ["generator", "discriminator"]:
            config.schedulers[name] = {
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 1,
                    "gamma": 1.0,
                },
                "interval": "step",
            }
        module = ESPnetLightningModule(WeightedGAN(), config)
        size = 2 if rank == 0 else 1
        samples = [
            (str(i), {"x": torch.tensor([float(rank + 1), 1.0]).numpy()})
            for i in range(size * 2)
        ]
        loader = torch.utils.data.DataLoader(
            samples, batch_size=size, collate_fn=CommonCollateFn()
        )
        trainer = lightning.Trainer(
            accelerator="cpu",
            devices=2,
            strategy="ddp_find_unused_parameters_true",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=0,
            use_distributed_sampler=False,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            default_root_dir=root,
        )
        trainer.fit(module, train_dataloaders=loader)
        queue.put(
            (
                rank,
                torch.cat([p.detach().flatten() for p in module.parameters()]).tolist(),
            )
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.execution_timeout(90)
def test_two_rank_manual_optimizer_gradients(tmp_path):
    queue = mp.get_context("fork").Queue()
    mp.start_processes(
        _manual_worker,
        args=(str(tmp_path), queue),
        nprocs=2,
        start_method="fork",
        join=True,
    )
    results = dict(queue.get(timeout=5) for _ in range(2))
    torch.testing.assert_close(torch.tensor(results[0]), torch.tensor(results[1]))
    queue.close()
