"""Evaluate a validation-selected CIFAR-10 checkpoint and draw report figures.

The official test split is used only here.  Do not choose hyperparameters from
these results. Per-class ``accuracy`` means recall: correctly classified items
of a true class divided by the number of test items in that class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import build_model


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available.")
    return torch.device(requested)


def read_history(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def draw_curves(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.15), constrained_layout=True)
    for key, label, color in (("train", "Training", "#2474a5"), ("val", "Validation", "#d47727")):
        axes[0].plot(epochs, [row[key + "_loss"] for row in history], label=label, color=color, linewidth=1.7)
        accuracy = [row[key + "_accuracy"] for row in history]
        # The companion training implementation records accuracies in [0, 1].
        axes[1].plot(epochs, [value * 100 for value in accuracy], label=label, color=color, linewidth=1.7)
    for axis, title, ylabel in zip(axes, ("Cross-entropy loss", "Classification accuracy"), ("Loss", "Accuracy (%)")):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)


def draw_confusion(matrix: np.ndarray, path: Path) -> None:
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    figure, axis = plt.subplots(figsize=(7.1, 5.9), constrained_layout=True)
    mesh = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8,
                      color="white" if normalized[row, col] > 0.55 else "#183346")
    axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right", fontsize=9)
    axis.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
    axis.set(xlabel="Predicted class", ylabel="True class", title="Test confusion matrix (counts; color = row fraction)")
    figure.colorbar(mesh, ax=axis, label="Fraction of true class", fraction=0.046, pad=0.03)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)


def choose_mistakes(labels: np.ndarray, predictions: np.ndarray, confidence: np.ndarray, count: int = 12) -> list[int]:
    wrong = np.flatnonzero(labels != predictions)
    return sorted(wrong.tolist(), key=lambda index: (-float(confidence[index]), index))[:count]


def summarize_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Return standard multi-class metrics; each row is a true class.

    This helper is independent of the dataset and can be checked against small
    hand-computable examples. Empty predicted classes have precision zero.
    """
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError("labels and predictions must be equally sized, nonempty 1-D arrays")
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("class identifiers must be integers")
    if min(int(labels.min()), int(predictions.min())) < 0 or max(int(labels.max()), int(predictions.max())) >= len(CLASSES):
        raise ValueError("class identifiers must be between 0 and 9")
    matrix = np.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    per_class = []
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class.append({"class_id": index, "class_name": name, "support": support, "correct": tp,
                          "accuracy": recall, "precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return {"num_test_samples": len(labels), "correct": int(np.trace(matrix)),
            "overall_accuracy": float(np.trace(matrix) / len(labels)),
            "per_class_accuracy_definition": "Recall: correct predictions / true-class support. Not one-vs-rest accuracy.",
            "per_class": per_class, "macro_f1": float(np.mean([item["f1"] for item in per_class])),
            "class_names": list(CLASSES), "confusion_matrix": matrix.tolist()}


def draw_mistakes(dataset: datasets.CIFAR10, labels: np.ndarray, predictions: np.ndarray,
                  confidence: np.ndarray, indices: list[int], path: Path) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(9.3, 6.8), constrained_layout=True)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indices):
        axis.imshow(dataset.data[index], interpolation="nearest")
        axis.set_title(f"#{index} True: {CLASSES[labels[index]]}\nPred: {CLASSES[predictions[index]]} ({confidence[index]:.1%})", fontsize=9)
    title = "Most confident test errors (descriptive examples, not a random sample)"
    if not indices:
        title = "No misclassified test examples"
    figure.suptitle(title, fontsize=12)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)


def gradcam(model: nn.Module, sample: torch.Tensor, target: int) -> tuple[np.ndarray, bool]:
    """Compute Grad-CAM for a predicted logit without changing model weights."""
    # A model can expose a suitable final spatial layer; otherwise use its final
    # convolution. Hooks are removed even if differentiation fails.
    target_layer = getattr(model, "gradcam_layer", None)
    if target_layer is None:
        target_layer = [module for module in model.modules() if isinstance(module, nn.Conv2d)][-1]
    activations: list[torch.Tensor] = []
    handle = target_layer.register_forward_hook(lambda module, inputs, output: activations.append(output))
    try:
        # Requiring input gradients makes the function work for frozen models too.
        value = sample.detach().unsqueeze(0).requires_grad_(True)
        logits = model(value)
        activation = activations[-1]
        gradient = torch.autograd.grad(logits[0, target], activation)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heat = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heat = torch.nn.functional.interpolate(heat, size=(32, 32), mode="bilinear", align_corners=False)
        heat = heat.detach()[0, 0].float().cpu().numpy()
        maximum = float(heat.max())
        if maximum > 0:
            heat = heat / maximum
        return heat, maximum > 0
    finally:
        handle.remove()


def draw_gradcam(model: nn.Module, dataset: datasets.CIFAR10, labels: np.ndarray,
                 predictions: np.ndarray, confidence: np.ndarray, wrong_indices: list[int],
                 device: torch.device, path: Path) -> list[dict]:
    """Show three errors and three deterministic correct examples for context."""
    correct = np.flatnonzero(labels == predictions)
    correct_indices = correct[:3].tolist()
    selected = wrong_indices[:3] + correct_indices
    figure, axes = plt.subplots(2, 6, figsize=(10.0, 5.4), constrained_layout=True)
    for axis in axes.flat:
        axis.axis("off")
    entries = []
    model.eval()
    for col, index in enumerate(selected):
        heat, active = gradcam(model, dataset[index][0].to(device), int(predictions[index]))
        raw = dataset.data[index]
        axes[0, col].imshow(raw, interpolation="nearest")
        axes[0, col].set_title(f"#{index} {CLASSES[labels[index]]}\nPred: {CLASSES[predictions[index]]}", fontsize=10)
        axes[1, col].imshow(raw, interpolation="nearest")
        axes[1, col].imshow(heat, cmap="jet", vmin=0, vmax=1, alpha=0.45, interpolation="bilinear")
        axes[1, col].set_title("Predicted-class\nGrad-CAM" if active else "Zero positive\nactivation", fontsize=9.5)
        entries.append({"test_index": index, "true_class": CLASSES[labels[index]],
                        "predicted_class": CLASSES[predictions[index]],
                        "confidence": float(confidence[index]), "positive_cam": active})
    figure.suptitle("Grad-CAM: model evidence for its predicted class (3 errors / 3 correct examples)", fontsize=11)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    return entries


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.threads < 1:
        raise ValueError("batch-size and threads must be positive; num-workers must be nonnegative")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
    if len(dataset) != 10000:
        raise ValueError(f"Expected official 10,000-image test split, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    labels_list, predictions_list, confidences_list = [], [], []
    total_loss = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch, target in loader:
            batch, target = batch.to(device, non_blocking=True), target.to(device, non_blocking=True)
            logits = model(batch)
            total_loss += float(criterion(logits, target).item())
            probability, prediction = logits.softmax(dim=1).max(dim=1)
            labels_list.append(target.cpu().numpy())
            predictions_list.append(prediction.cpu().numpy())
            confidences_list.append(probability.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - start
    labels = np.concatenate(labels_list)
    predictions = np.concatenate(predictions_list)
    confidence = np.concatenate(confidences_list)
    prediction_metrics = summarize_predictions(labels, predictions)
    matrix = np.asarray(prediction_metrics["confusion_matrix"])
    wrong_indices = choose_mistakes(labels, predictions, confidence)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    metrics = {
        "model_name": checkpoint["model_name"], "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash, "selected_epoch": checkpoint.get("epoch"),
        "best_val_accuracy": checkpoint.get("best_val_accuracy"),
        **prediction_metrics, "loss": total_loss / len(dataset),
        "evaluation_seconds": evaluation_seconds,
        "device": str(device), "torch_version": torch.__version__, "python_version": platform.python_version(),
        "error_selection": "Up to 12 incorrect examples with highest predicted-class softmax probability; descriptive, not representative.",
        "top_confident_errors": [{"test_index": index, "true_class": CLASSES[labels[index]],
                                  "predicted_class": CLASSES[predictions[index]], "confidence": float(confidence[index])}
                                 for index in wrong_indices],
        "test_protocol": "Entire official test split, deterministic normalization; no test-time augmentation. Checkpoint selected only by validation accuracy.",
    }
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("test_index", "true_id", "true_class", "predicted_id", "predicted_class", "confidence", "correct"))
        for index, (label, prediction, probability) in enumerate(zip(labels, predictions, confidence)):
            writer.writerow((index, int(label), CLASSES[label], int(prediction), CLASSES[prediction], float(probability), int(label == prediction)))
    history = read_history(args.checkpoint.parent / "history.csv")
    draw_curves(history, args.output_dir / "training_curves.png")
    draw_confusion(matrix, args.output_dir / "confusion_matrix.png")
    draw_mistakes(dataset, labels, predictions, confidence, wrong_indices, args.output_dir / "misclassifications.png")
    metrics["gradcam_examples"] = draw_gradcam(model, dataset, labels, predictions, confidence, wrong_indices,
                                               device, args.output_dir / "gradcam.png")
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"test_accuracy": metrics["overall_accuracy"], "test_samples": len(dataset),
                      "macro_f1": metrics["macro_f1"], "evaluation_seconds": evaluation_seconds,
                      "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
