#!/usr/bin/env python3
"""
Ablation studies for emg2qwerty decoding performance.

Experiments:
  augmentation   - Compare data augmentation / pre-processing techniques
  channels       - Electrode channel count vs CER
  data_amount    - Training data fraction vs CER
  sampling_rate  - EMG sampling rate vs CER

Usage:
    python scripts/run_experiments.py --experiment augmentation
    python scripts/run_experiments.py --experiment channels
    python scripts/run_experiments.py --experiment data_amount
    python scripts/run_experiments.py --experiment sampling_rate
    python scripts/run_experiments.py --experiment all
    python scripts/run_experiments.py --experiment all --max_epochs 30 --model gru
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

# ---------------------------------------------------------------------------
# Ensure the repo root is on the path so emg2qwerty can be imported
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emg2qwerty.data import WindowedEMGDataset  # noqa: E402
from emg2qwerty.decoder import CTCGreedyDecoder  # noqa: E402
from emg2qwerty.lightning import (  # noqa: E402
    GRUCTCModule,
    TDSConvCTCModule,
    WindowedEMGDataModule,
)
from emg2qwerty.transforms import (  # noqa: E402
    ChannelSubset,
    Compose,
    Downsample,
    ForEach,
    GaussianNoise,
    LogSpectrogram,
    RandomBandRotation,
    SpecAugment,
    TemporalAlignmentJitter,
    TimeWarp,
    ToTensor,
)

# ---------------------------------------------------------------------------
# Default session split for user 89335547
# ---------------------------------------------------------------------------
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


def session_paths(sessions: list[str]) -> list[Path]:
    return [DATA_ROOT / f"{s}.hdf5" for s in sessions]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
OPTIMIZER_CFG = OmegaConf.create(
    {"_target_": "torch.optim.Adam", "lr": 1e-3}
)

DECODER_CFG = OmegaConf.create(
    {"_target_": "emg2qwerty.decoder.CTCGreedyDecoder"}
)


def lr_scheduler_cfg(max_epochs: int) -> DictConfig:
    return OmegaConf.create(
        {
            "scheduler": {
                "_target_": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR",
                "warmup_epochs": min(5, max_epochs // 3),
                "max_epochs": max_epochs,
                "warmup_start_lr": 1e-8,
                "eta_min": 1e-6,
            },
            "interval": "epoch",
        }
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_module(
    model_type: str,
    in_features: int = 528,
    electrode_channels: int = 16,
    max_epochs: int = 50,
) -> pl.LightningModule:
    """Build a TDS-Conv or GRU CTC module with the given parameters.

    Creates a dynamic subclass so ELECTRODE_CHANNELS can be overridden
    without mutating the base class.
    """
    sched = lr_scheduler_cfg(max_epochs)

    if model_type == "tds":
        cls = type(
            "_TDS",
            (TDSConvCTCModule,),
            {"ELECTRODE_CHANNELS": electrode_channels},
        )
        return cls(
            in_features=in_features,
            mlp_features=[384],
            block_channels=[24, 24, 24, 24],
            kernel_width=32,
            optimizer=OPTIMIZER_CFG,
            lr_scheduler=sched,
            decoder=DECODER_CFG,
        )
    else:
        cls = type(
            "_GRU",
            (GRUCTCModule,),
            {"ELECTRODE_CHANNELS": electrode_channels},
        )
        return cls(
            in_features=in_features,
            mlp_features=[384],
            hidden_size=384,
            num_layers=2,
            dropout=0.2,
            optimizer=OPTIMIZER_CFG,
            lr_scheduler=sched,
            decoder=DECODER_CFG,
        )


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
def train_and_evaluate(
    module: pl.LightningModule,
    train_sessions: list[Path],
    val_sessions: list[Path],
    test_sessions: list[Path],
    train_transform: Any,
    val_transform: Any,
    max_epochs: int = 50,
    batch_size: int = 32,
    window_length: int = 8000,
    padding: tuple[int, int] = (1800, 200),
    num_workers: int = 4,
    accelerator: str = "auto",
    run_label: str = "experiment",
    precision: int = 16,
    early_stop_patience: int = 8,
) -> dict[str, float]:
    """Run a full train / val / test cycle and return the final CER metrics."""

    # Skip if results already exist for this run
    cached_results_path = REPO_ROOT / "experiment_logs" / run_label / "results.json"
    if cached_results_path.exists():
        print(f"\n  [SKIP] Found cached results for '{run_label}', loading...")
        with open(cached_results_path) as f:
            metrics = json.load(f)
        print(f"  Cached val CER: {metrics.get('val_val/CER', 'N/A')}, "
              f"test CER: {metrics.get('test_test/CER', 'N/A')}")
        return metrics

    pl.seed_everything(1501, workers=True)

    datamodule = WindowedEMGDataModule(
        window_length=window_length,
        padding=padding,
        batch_size=batch_size,
        num_workers=num_workers,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        test_sessions=test_sessions,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
    )

    output_dir = REPO_ROOT / "experiment_logs" / run_label
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
            patience=early_stop_patience,
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(),
    ]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        precision=precision,
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
        best_module = module.load_from_checkpoint(best_path)
    else:
        best_module = module

    val_results = trainer.validate(best_module, datamodule)
    test_results = trainer.test(best_module, datamodule)

    # Free resources between experiment runs
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    metrics = {}
    if val_results:
        metrics.update({f"val_{k}": v for k, v in val_results[0].items()})
    if test_results:
        metrics.update({f"test_{k}": v for k, v in test_results[0].items()})

    # Cache results so we can resume without re-training
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results for: {run_label}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}\n")

    return metrics


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------
def default_train_transform(
    n_fft: int = 64,
    hop_length: int = 16,
) -> Compose:
    return Compose(
        [
            ToTensor(),
            ForEach(RandomBandRotation(offsets=[-1, 0, 1])),
            TemporalAlignmentJitter(max_offset=120),
            LogSpectrogram(n_fft=n_fft, hop_length=hop_length),
            SpecAugment(
                n_time_masks=3,
                time_mask_param=25,
                n_freq_masks=2,
                freq_mask_param=4,
            ),
        ]
    )


def default_val_transform(
    n_fft: int = 64,
    hop_length: int = 16,
) -> Compose:
    return Compose(
        [
            ToTensor(),
            LogSpectrogram(n_fft=n_fft, hop_length=hop_length),
        ]
    )


# ===================================================================
# EXPERIMENT 2: Data Augmentation / Pre-processing Techniques
# ===================================================================
def run_augmentation_experiment(args: argparse.Namespace) -> dict:
    """Compare different augmentation / pre-processing pipelines."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Data Augmentation / Pre-processing Techniques")
    print("=" * 70)

    train_paths = session_paths(TRAIN_SESSIONS)
    val_paths = session_paths([VAL_SESSION])
    test_paths = session_paths([TEST_SESSION])

    logspec = LogSpectrogram(n_fft=64, hop_length=16)
    specaug = SpecAugment(
        n_time_masks=3, time_mask_param=25,
        n_freq_masks=2, freq_mask_param=4,
    )
    band_rot = ForEach(RandomBandRotation(offsets=[-1, 0, 1]))
    temp_jitter = TemporalAlignmentJitter(max_offset=120)

    augmentation_configs = {
        "no_augmentation": Compose([
            ToTensor(),
            logspec,
        ]),
        "specaug_only": Compose([
            ToTensor(),
            logspec,
            specaug,
        ]),
        "default_all": Compose([
            ToTensor(),
            band_rot,
            temp_jitter,
            logspec,
            specaug,
        ]),
        "gaussian_noise": Compose([
            ToTensor(),
            GaussianNoise(std=0.05),
            band_rot,
            temp_jitter,
            logspec,
            specaug,
        ]),
        "time_warp": Compose([
            ToTensor(),
            TimeWarp(max_warp=50),
            band_rot,
            temp_jitter,
            logspec,
            specaug,
        ]),
        "kitchen_sink": Compose([
            ToTensor(),
            GaussianNoise(std=0.05),
            TimeWarp(max_warp=30),
            band_rot,
            temp_jitter,
            logspec,
            SpecAugment(
                n_time_masks=5, time_mask_param=40,
                n_freq_masks=3, freq_mask_param=6,
            ),
        ]),
    }

    val_transform = default_val_transform()
    results = {}

    for name, train_transform in augmentation_configs.items():
        print(f"\n--- Augmentation config: {name} ---")
        module = build_module(
            args.model, max_epochs=args.max_epochs,
        )
        metrics = train_and_evaluate(
            module=module,
            train_sessions=train_paths,
            val_sessions=val_paths,
            test_sessions=test_paths,
            train_transform=train_transform,
            val_transform=val_transform,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            run_label=f"augmentation/{name}",
        )
        results[name] = metrics

    _save_and_plot_augmentation(results)
    return results


def _save_and_plot_augmentation(results: dict) -> None:
    out_dir = REPO_ROOT / "experiment_results"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "augmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    names = list(results.keys())
    val_cers = [results[n].get("val_val/CER", results[n].get("val/CER", 0)) for n in names]
    test_cers = [results[n].get("test_test/CER", results[n].get("test/CER", 0)) for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, val_cers, width, label="Val CER", color="#4C72B0")
    ax.bar(x + width / 2, test_cers, width, label="Test CER", color="#DD8452")
    ax.set_xlabel("Augmentation Config")
    ax.set_ylabel("CER (%)")
    ax.set_title("Effect of Data Augmentation on Character Error Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "augmentation_results.png", dpi=150)
    plt.close()
    print(f"Augmentation results saved to {out_dir}")


# ===================================================================
# EXPERIMENT 3: Number of Electrode Channels vs CER
# ===================================================================
def run_channels_experiment(args: argparse.Namespace) -> dict:
    """Investigate the relationship between number of electrode channels and CER."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Number of Electrode Channels vs CER")
    print("=" * 70)

    train_paths = session_paths(TRAIN_SESSIONS)
    val_paths = session_paths([VAL_SESSION])
    test_paths = session_paths([TEST_SESSION])

    channel_counts = [1, 2, 4, 8, 16]
    n_fft = 64
    freq_bins = n_fft // 2 + 1  # 33

    results = {}
    for n_ch in channel_counts:
        print(f"\n--- Channels: {n_ch} / 16 ---")

        in_features = freq_bins * n_ch

        # No augmentation — isolate the effect of channel count
        train_transform = Compose([
            ToTensor(),
            ChannelSubset(n_channels=n_ch),
            LogSpectrogram(n_fft=n_fft, hop_length=16),
        ])
        val_transform = Compose([
            ToTensor(),
            ChannelSubset(n_channels=n_ch),
            LogSpectrogram(n_fft=n_fft, hop_length=16),
        ])

        module = build_module(
            args.model,
            in_features=in_features,
            electrode_channels=n_ch,
            max_epochs=args.max_epochs,
        )
        metrics = train_and_evaluate(
            module=module,
            train_sessions=train_paths,
            val_sessions=val_paths,
            test_sessions=test_paths,
            train_transform=train_transform,
            val_transform=val_transform,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            run_label=f"channels/{n_ch}ch",
        )
        results[n_ch] = metrics

    _save_and_plot_channels(results)
    return results


def _save_and_plot_channels(results: dict) -> None:
    out_dir = REPO_ROOT / "experiment_results"
    out_dir.mkdir(exist_ok=True)

    serializable = {str(k): v for k, v in results.items()}
    with open(out_dir / "channels_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    channels = sorted(results.keys())
    val_cers = [results[c].get("val_val/CER", results[c].get("val/CER", 0)) for c in channels]
    test_cers = [results[c].get("test_test/CER", results[c].get("test/CER", 0)) for c in channels]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(channels, val_cers, "o-", label="Val CER", color="#4C72B0", linewidth=2, markersize=8)
    ax.plot(channels, test_cers, "s--", label="Test CER", color="#DD8452", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Electrode Channels (per wrist)")
    ax.set_ylabel("CER (%)")
    ax.set_title("Effect of Number of Electrode Channels on CER")
    ax.set_xticks(channels)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "channels_results.png", dpi=150)
    plt.close()
    print(f"Channel results saved to {out_dir}")


# ===================================================================
# EXPERIMENT 4: Amount of Training Data vs CER
# ===================================================================
def run_data_amount_experiment(args: argparse.Namespace) -> dict:
    """Investigate the relationship between amount of training data and CER."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Amount of Training Data vs CER")
    print("=" * 70)

    val_paths = session_paths([VAL_SESSION])
    test_paths = session_paths([TEST_SESSION])
    all_train = list(TRAIN_SESSIONS)

    fractions = [0.125, 0.25, 0.5, 1.0]
    # No augmentation — isolate the effect of data quantity
    train_transform = default_val_transform()
    val_transform = default_val_transform()

    results = {}
    for frac in fractions:
        n_sessions = max(1, int(len(all_train) * frac))
        selected = all_train[:n_sessions]
        train_paths = session_paths(selected)

        label = f"{frac*100:.1f}% ({n_sessions} sessions)"
        print(f"\n--- Training data: {label} ---")

        module = build_module(
            args.model, max_epochs=args.max_epochs,
        )
        metrics = train_and_evaluate(
            module=module,
            train_sessions=train_paths,
            val_sessions=val_paths,
            test_sessions=test_paths,
            train_transform=train_transform,
            val_transform=val_transform,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            run_label=f"data_amount/{n_sessions}_sessions",
        )
        results[frac] = {
            "n_sessions": n_sessions,
            "metrics": metrics,
        }

    _save_and_plot_data_amount(results)
    return results


def _save_and_plot_data_amount(results: dict) -> None:
    out_dir = REPO_ROOT / "experiment_results"
    out_dir.mkdir(exist_ok=True)

    serializable = {str(k): v for k, v in results.items()}
    with open(out_dir / "data_amount_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    fracs = sorted(results.keys())
    n_sessions = [results[f]["n_sessions"] for f in fracs]
    val_cers = [
        results[f]["metrics"].get("val_val/CER", results[f]["metrics"].get("val/CER", 0))
        for f in fracs
    ]
    test_cers = [
        results[f]["metrics"].get("test_test/CER", results[f]["metrics"].get("test/CER", 0))
        for f in fracs
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_sessions, val_cers, "o-", label="Val CER", color="#4C72B0", linewidth=2, markersize=8)
    ax.plot(n_sessions, test_cers, "s--", label="Test CER", color="#DD8452", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Training Sessions")
    ax.set_ylabel("CER (%)")
    ax.set_title("Effect of Training Data Amount on CER")

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    pct_ticks = [f"{f*100:.0f}%" for f in fracs]
    ax2.set_xticks(n_sessions)
    ax2.set_xticklabels(pct_ticks)
    ax2.set_xlabel("Fraction of Training Data")

    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "data_amount_results.png", dpi=150)
    plt.close()
    print(f"Data amount results saved to {out_dir}")


# ===================================================================
# EXPERIMENT 5: Sampling Rate vs CER
# ===================================================================
def run_sampling_rate_experiment(args: argparse.Namespace) -> dict:
    """Investigate the relationship between sampling rate and CER."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Sampling Rate vs CER")
    print("=" * 70)

    train_paths = session_paths(TRAIN_SESSIONS)
    val_paths = session_paths([VAL_SESSION])
    test_paths = session_paths([TEST_SESSION])

    # Original: 2000 Hz, n_fft=64, hop_length=16
    # Downsample factor k -> rate=2000/k, n_fft=64/k, hop=16/k
    # freq_bins = n_fft/k / 2 + 1
    configs = [
        {"factor": 1, "rate_hz": 2000, "n_fft": 64, "hop": 16},
        {"factor": 2, "rate_hz": 1000, "n_fft": 32, "hop": 8},
        {"factor": 4, "rate_hz": 500,  "n_fft": 16, "hop": 4},
        {"factor": 8, "rate_hz": 250,  "n_fft": 8,  "hop": 2},
    ]

    results = {}
    for cfg in configs:
        factor = cfg["factor"]
        rate = cfg["rate_hz"]
        n_fft = cfg["n_fft"]
        hop = cfg["hop"]
        freq_bins = n_fft // 2 + 1
        in_features = freq_bins * 16

        print(f"\n--- Sampling rate: {rate} Hz (downsample x{factor}) ---")
        print(f"    n_fft={n_fft}, hop={hop}, freq_bins={freq_bins}, "
              f"in_features={in_features}")

        # No augmentation — isolate the effect of sampling rate
        transforms_list = [ToTensor()]
        if factor > 1:
            transforms_list.append(Downsample(factor=factor))
        transforms_list.append(LogSpectrogram(n_fft=n_fft, hop_length=hop))
        train_transform = Compose(transforms_list)

        val_list = [ToTensor()]
        if factor > 1:
            val_list.append(Downsample(factor=factor))
        val_list.append(LogSpectrogram(n_fft=n_fft, hop_length=hop))
        val_transform = Compose(val_list)

        module = build_module(
            args.model,
            in_features=in_features,
            max_epochs=args.max_epochs,
        )
        metrics = train_and_evaluate(
            module=module,
            train_sessions=train_paths,
            val_sessions=val_paths,
            test_sessions=test_paths,
            train_transform=train_transform,
            val_transform=val_transform,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            run_label=f"sampling_rate/{rate}Hz",
        )
        results[rate] = {
            "downsample_factor": factor,
            "n_fft": n_fft,
            "hop_length": hop,
            "freq_bins": freq_bins,
            "in_features": in_features,
            "metrics": metrics,
        }

    _save_and_plot_sampling_rate(results)
    return results


def _save_and_plot_sampling_rate(results: dict) -> None:
    out_dir = REPO_ROOT / "experiment_results"
    out_dir.mkdir(exist_ok=True)

    serializable = {str(k): v for k, v in results.items()}
    with open(out_dir / "sampling_rate_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    rates = sorted(results.keys())
    val_cers = [
        results[r]["metrics"].get("val_val/CER", results[r]["metrics"].get("val/CER", 0))
        for r in rates
    ]
    test_cers = [
        results[r]["metrics"].get("test_test/CER", results[r]["metrics"].get("test/CER", 0))
        for r in rates
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rates, val_cers, "o-", label="Val CER", color="#4C72B0", linewidth=2, markersize=8)
    ax.plot(rates, test_cers, "s--", label="Test CER", color="#DD8452", linewidth=2, markersize=8)
    ax.set_xlabel("Sampling Rate (Hz)")
    ax.set_ylabel("CER (%)")
    ax.set_title("Effect of EMG Sampling Rate on CER")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r} Hz" for r in rates])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sampling_rate_results.png", dpi=150)
    plt.close()
    print(f"Sampling rate results saved to {out_dir}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run emg2qwerty ablation experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["augmentation", "channels", "data_amount", "sampling_rate", "all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--model", type=str, default="tds", choices=["tds", "gru"],
        help="Model architecture (default: tds)",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=40,
        help="Maximum training epochs per run (default: 40)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        help="PyTorch Lightning accelerator (default: auto)",
    )
    args = parser.parse_args()

    all_results = {}
    experiments = {
        "augmentation": run_augmentation_experiment,
        "channels": run_channels_experiment,
        "data_amount": run_data_amount_experiment,
        "sampling_rate": run_sampling_rate_experiment,
    }

    if args.experiment == "all":
        for name, fn in experiments.items():
            all_results[name] = fn(args)
    else:
        all_results[args.experiment] = experiments[args.experiment](args)

    out_dir = REPO_ROOT / "experiment_results"
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "all_results.json"

    def _make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        return obj

    with open(summary_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)

    print(f"\nAll results saved to {summary_path}")
    print("Plots saved to experiment_results/")


if __name__ == "__main__":
    main()
