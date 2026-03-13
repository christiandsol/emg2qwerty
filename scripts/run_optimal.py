#!/usr/bin/env python3
"""
Optimal run: combines the best val-CER settings from each ablation.

Settings:
  - No augmentation (val CER winner)
  - 16 channels (val CER winner — full set)
  - 16 sessions / 100% data (val CER winner — all data)
  - 1000 Hz / downsample x2 (val CER winner)
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emg2qwerty.lightning import TDSConvCTCModule, WindowedEMGDataModule
from emg2qwerty.transforms import Compose, Downsample, LogSpectrogram, ToTensor

DATA_ROOT = Path("/mnt/data/key_data")

TRAIN_SESSIONS = [
    "2021-06-03-1622765527-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-02-1622681518-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-04-1622863166-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-22-1627003020-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-21-1626916256-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-22-1627004019-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-05-1622885888-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-03-1622764398-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-21-1626917264-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-05-1622889105-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-03-1622766673-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-04-1622861066-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-22-1627001995-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-06-05-1622884635-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
    "2021-07-21-1626915176-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f",
]
VAL_SESSION = "2021-06-04-1622862148-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"
TEST_SESSION = "2021-06-02-1622682789-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"


def main():
    pl.seed_everything(1501, workers=True)

    # 1000 Hz = downsample x2, n_fft=32, hop=8 => freq_bins=17, in_features=17*16=272
    n_fft = 32
    hop_length = 8
    freq_bins = n_fft // 2 + 1  # 17
    in_features = freq_bins * 16  # 272

    transform = Compose([
        ToTensor(),
        Downsample(factor=2),
        LogSpectrogram(n_fft=n_fft, hop_length=hop_length),
    ])

    train_paths = [DATA_ROOT / f"{s}.hdf5" for s in TRAIN_SESSIONS]
    val_paths = [DATA_ROOT / f"{VAL_SESSION}.hdf5"]
    test_paths = [DATA_ROOT / f"{TEST_SESSION}.hdf5"]

    datamodule = WindowedEMGDataModule(
        window_length=8000,
        padding=(1800, 200),
        batch_size=32,
        num_workers=2,
        train_sessions=train_paths,
        val_sessions=val_paths,
        test_sessions=test_paths,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
    )

    optimizer_cfg = OmegaConf.create({"_target_": "torch.optim.Adam", "lr": 1e-3})
    lr_scheduler_cfg = OmegaConf.create({
        "scheduler": {
            "_target_": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR",
            "warmup_epochs": 5,
            "max_epochs": 40,
            "warmup_start_lr": 1e-8,
            "eta_min": 1e-6,
        },
        "interval": "epoch",
    })
    decoder_cfg = OmegaConf.create({"_target_": "emg2qwerty.decoder.CTCGreedyDecoder"})

    module = TDSConvCTCModule(
        in_features=in_features,
        mlp_features=[384],
        block_channels=[24, 24, 24, 24],
        kernel_width=32,
        optimizer=optimizer_cfg,
        lr_scheduler=lr_scheduler_cfg,
        decoder=decoder_cfg,
    )

    output_dir = REPO_ROOT / "experiment_logs" / "optimal_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            monitor="val/CER",
            mode="min",
            save_last=True,
            verbose=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/CER",
            mode="min",
            patience=8,
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(),
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=40,
        precision=16,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=2,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(module, datamodule)

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        best_module = TDSConvCTCModule.load_from_checkpoint(best_path)
    else:
        best_module = module

    val_results = trainer.validate(best_module, datamodule)
    test_results = trainer.test(best_module, datamodule)

    metrics = {}
    if val_results:
        metrics.update({f"val_{k}": v for k, v in val_results[0].items()})
    if test_results:
        metrics.update({f"test_{k}": v for k, v in test_results[0].items()})

    with open(output_dir / "results.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"\n{'='*60}")
    print("OPTIMAL RUN RESULTS")
    print(f"Config: no_augmentation + 16ch + 16 sessions + 1000Hz")
    print(f"{'='*60}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
