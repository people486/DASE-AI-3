"""Train on CIFAR-10's training split; select checkpoints using validation only."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .models import build_model

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
HISTORY_FIELDS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr", "seconds"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses every training example")
    parser.add_argument("--val-limit", type=int, default=0, help="0 uses the full validation split")
    parser.add_argument("--augment", choices=["none", "standard"], default="standard")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--resume", type=Path, help="Resume from last.pt; keep the original configuration")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PyTorch algorithms")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "threads"):
        if getattr(args, name) < 1:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0 or args.train_limit < 0 or args.val_limit < 0 or args.patience < 0:
        parser.error("Worker counts, data limits, and patience must be nonnegative")
    if not 0 < args.val_size < 50000:
        parser.error("--val-size must be between 1 and 49999")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative")
    return args


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    # DataLoader derives this seed from its saved generator state.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stratified_take(indices, labels, count, seed):
    """Return disjoint, reproducible selected/remainder lists, preserving class balance."""
    if count < 0 or count > len(indices):
        raise ValueError("Requested {} examples from {} available".format(count, len(indices)))
    if not len(indices):
        return [], []
    generator = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    classes, sizes = np.unique(labels[indices], return_counts=True)
    ideal = sizes * (count / len(indices))
    allocations = np.floor(ideal).astype(int)
    remainder = count - int(allocations.sum())
    order = np.argsort(-(ideal - allocations), kind="stable")
    allocations[order[:remainder]] += 1
    selected, remaining = [], []
    for label, allocation in zip(classes, allocations):
        pool = indices[labels[indices] == label].copy()
        generator.shuffle(pool)
        selected.extend(pool[:allocation].tolist())
        remaining.extend(pool[allocation:].tolist())
    generator.shuffle(selected)
    generator.shuffle(remaining)
    return selected, remaining


def indices_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def build_loaders(args, device):
    training_steps = []
    if args.augment == "standard":
        training_steps.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    training_steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    clean_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    # Separate dataset instances prevent augmentations from being used for validation.
    training_data = datasets.CIFAR10(str(args.data_dir), train=True, download=True,
                                     transform=transforms.Compose(training_steps))
    validation_data = datasets.CIFAR10(str(args.data_dir), train=True, download=False,
                                       transform=clean_transform)
    labels = training_data.targets
    validation_indices, training_indices = stratified_take(
        list(range(len(labels))), labels, args.val_size, args.seed
    )
    if args.train_limit:
        training_indices, _ = stratified_take(training_indices, labels, args.train_limit, args.seed + 1)
    if args.val_limit:
        validation_indices, _ = stratified_take(validation_indices, labels, args.val_limit, args.seed + 2)
    assert not set(training_indices).intersection(validation_indices)
    loader_generator = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    # Restart worker processes each epoch so saved generator state also reproduces augmentation RNG.
    training_loader = DataLoader(Subset(training_data, training_indices), shuffle=True,
                                 generator=loader_generator, **common)
    validation_loader = DataLoader(Subset(validation_data, validation_indices), shuffle=False, **common)
    split = {
        "train_indices": training_indices, "val_indices": validation_indices,
        "train_sha256": indices_hash(training_indices), "val_sha256": indices_hash(validation_indices),
        "train_size": len(training_indices), "val_size": len(validation_indices),
        "train_class_counts": np.bincount(np.asarray(labels)[training_indices], minlength=10).tolist(),
        "val_class_counts": np.bincount(np.asarray(labels)[validation_indices], minlength=10).tolist(),
        "test_used_for_training": False,
    }
    return training_loader, validation_loader, loader_generator, split


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              amp_enabled=False, channels_last=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, samples = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(images)
                loss = criterion(predictions, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (predictions.argmax(1) == labels).sum().item()
            samples += count
    return total_loss / samples, correct / samples


def json_config(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed, args.deterministic)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device("cuda" if (args.device == "cuda" or args.device == "auto"
                                     and torch.cuda.is_available()) else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.csv"
    if not args.resume and (history_path.exists() or (args.output_dir / "best.pt").exists()):
        raise FileExistsError("Output contains a previous run. Use another output directory or --resume last.pt")
    train_loader, val_loader, loader_generator, split = build_loaders(args, device)
    model = build_model(args.model).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_accuracy, best_epoch, stale_epochs = 1, -1.0, 0, 0
    accumulated_seconds = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        previous = checkpoint["config"]
        for field in ("model", "epochs", "batch_size", "seed", "val_size", "train_limit", "val_limit",
                      "augment", "optimizer", "lr", "weight_decay", "label_smoothing", "num_workers",
                      "no_amp", "channels_last", "deterministic"):
            if previous.get(field) != getattr(args, field):
                raise ValueError("Resume configuration differs for {}: {} vs {}".format(
                    field, previous.get(field), getattr(args, field)))
        if checkpoint["split_hashes"] != [split["train_sha256"], split["val_sha256"]]:
            raise ValueError("Dataset split changed during resume")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_val_accuracy"]
        best_epoch = checkpoint["best_epoch"]
        stale_epochs = checkpoint["stale_epochs"]
        accumulated_seconds = checkpoint["training_seconds"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        loader_generator.set_state(checkpoint["loader_rng"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["cuda_rng"]])
        if not history_path.exists():
            raise FileNotFoundError("Resume requires history.csv in the original output directory")
        with history_path.open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        if not history or int(history[-1]["epoch"]) != checkpoint["epoch"]:
            raise ValueError("history.csv does not end at the checkpoint's epoch")
    config = json_config(args)
    save_json(args.output_dir / "config.json", config)
    save_json(args.output_dir / "split_indices.json", split)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    environment = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda, "amp": amp_enabled,
    }
    print(json.dumps({"model": args.model, "parameters": parameter_count,
                      "train_size": split["train_size"], "val_size": split["val_size"],
                      "device": str(device), "amp": amp_enabled}, ensure_ascii=False), flush=True)
    total_started = time.perf_counter()
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer,
                                               scaler, amp_enabled, args.channels_last)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device,
                                           amp_enabled=amp_enabled, channels_last=args.channels_last)
        scheduler.step()
        elapsed = time.perf_counter() - started
        improved = val_accuracy > best_accuracy
        if improved:
            best_accuracy, best_epoch, stale_epochs = val_accuracy, epoch, 0
        else:
            stale_epochs += 1
        final_epoch = epoch
        row = dict(zip(HISTORY_FIELDS, [epoch, train_loss, train_accuracy, val_loss,
                                        val_accuracy, learning_rate, elapsed]))
        with history_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = {
            "model_name": args.model, "model_state": model.state_dict(), "config": config,
            "epoch": epoch, "best_val_accuracy": best_accuracy, "best_epoch": best_epoch,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "stale_epochs": stale_epochs,
            "training_seconds": accumulated_seconds + time.perf_counter() - total_started,
            "split_hashes": [split["train_sha256"], split["val_sha256"]],
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "loader_rng": loader_generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        # Atomic replacement prevents an interrupted write from destroying the previous checkpoint.
        for filename in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            temporary = args.output_dir / (filename + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, args.output_dir / filename)
        summary = {
            "model_name": args.model, "parameters": parameter_count, "config": config,
            "best_epoch": best_epoch, "best_val_accuracy": best_accuracy, "completed_epochs": epoch,
            "training_seconds": checkpoint["training_seconds"], "environment": environment,
            "split": {key: value for key, value in split.items() if not key.endswith("_indices")},
            "selection_rule": "Highest validation accuracy; ties retain the earlier epoch",
            "stopped_early": bool(args.patience and stale_epochs >= args.patience),
        }
        save_json(args.output_dir / "summary.json", summary)
        print("epoch {}/{} train loss={:.4f} acc={:.4f} val loss={:.4f} acc={:.4f} lr={:.6f} {:.1f}s{}".format(
            epoch, args.epochs, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate,
            elapsed, " *best" if improved else ""), flush=True)
        if args.patience and stale_epochs >= args.patience:
            break
    print("Finished at epoch {}; best validation accuracy {:.4f} at epoch {}. Test set was not accessed.".format(
        final_epoch, best_accuracy, best_epoch), flush=True)


if __name__ == "__main__":
    main()
