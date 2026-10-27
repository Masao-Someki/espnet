"""Final validation checkpoints participate in the produced top-K average."""

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from espnet3.components.callbacks.default_callbacks import AverageCheckpointsCallback


class _Scalar(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight

    def validation_step(self, batch, batch_idx):
        self.log("valid/loss", self.weight, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_final_average_includes_all_best_checkpoints(tmp_path):
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="valid/loss",
        save_top_k=3,
        save_on_train_epoch_end=True,
    )
    average = AverageCheckpointsCallback(tmp_path, [checkpoint])
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint, average],
    )
    loader = torch.utils.data.DataLoader([torch.ones(1)])
    trainer.fit(_Scalar(), train_dataloaders=loader, val_dataloaders=loader)
    actual = torch.load(tmp_path / "valid.loss.ave_3best.pth", weights_only=False)[
        "weight"
    ]
    expected = torch.stack(
        [
            torch.load(path, weights_only=False)["state_dict"]["weight"]
            for path in checkpoint.best_k_models
        ]
    ).mean()
    torch.testing.assert_close(actual, expected)
