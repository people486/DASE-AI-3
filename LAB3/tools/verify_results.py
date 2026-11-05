"""Independently cross-check the three completed CIFAR-10 experiment records.

This script rebuilds metrics from prediction rows using only the standard
library, without importing the training/evaluation metric implementations.
If PyTorch and local best.pt files are present, it also inspects those locally
created checkpoints. Only load checkpoints produced by your own experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import struct
from datetime import datetime, timezone
from pathlib import Path


CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
GROUPS = {
    "baseline": ("baseline", "none"),
    "ablation": ("baseline", "standard"),
    "improved": ("improved", "standard"),
}
REPOSITORY = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices):
    """Hash signed little-endian int64 indices, independently of NumPy."""
    digest = hashlib.sha256()
    for offset in range(0, len(indices), 4096):
        part = indices[offset:offset + 4096]
        digest.update(struct.pack("<{}q".format(len(part)), *part))
    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def finite_number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            "{} is not a number".format(label))
    value = float(value)
    require(math.isfinite(value), "{} is not finite".format(label))
    return value


def equal_number(actual, expected, label, tolerance=1e-12):
    actual = finite_number(actual, label)
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
            "{} differs: {} != {}".format(label, actual, expected))


def check_predictions(directory, metrics):
    matrix = [[0] * 10 for _ in range(10)]
    indices = set()
    required_columns = {
        "test_index", "true_id", "true_class", "predicted_id", "predicted_class", "confidence", "correct",
    }
    true_labels = bytearray(10000)
    wrong_rows = []
    with (directory / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(required_columns.issubset(reader.fieldnames or []), "Prediction CSV columns are incomplete")
        for line, row in enumerate(reader, 2):
            index, label, prediction, correct = (
                int(row[key]) for key in ("test_index", "true_id", "predicted_id", "correct")
            )
            require(0 <= index < 10000 and index not in indices,
                    "Invalid or duplicate test index at prediction line {}".format(line))
            require(0 <= label < 10 and 0 <= prediction < 10,
                    "Invalid class identifier at prediction line {}".format(line))
            require(row["true_class"] == CLASSES[label] and row["predicted_class"] == CLASSES[prediction],
                    "Class name/identifier mismatch at prediction line {}".format(line))
            require(correct == int(label == prediction), "Incorrect correctness flag at prediction line {}".format(line))
            confidence = finite_number(float(row["confidence"]), "prediction confidence")
            require(0.0 <= confidence <= 1.0, "Confidence is outside [0, 1]")
            indices.add(index)
            matrix[label][prediction] += 1
            true_labels[index] = label
            if label != prediction:
                wrong_rows.append((index, label, prediction, confidence))
    require(indices == set(range(10000)), "Predictions must cover every official test index exactly once")
    require(metrics["class_names"] == list(CLASSES), "Unexpected test class order")
    require(metrics["num_test_samples"] == 10000, "Metric sample count is not 10000")
    require(metrics["confusion_matrix"] == matrix, "Confusion matrix differs from prediction rows")
    support = [sum(row) for row in matrix]
    require(support == [1000] * 10, "Each true class must have 1000 official test examples")
    correct_count = sum(matrix[index][index] for index in range(10))
    require(metrics["correct"] == correct_count, "Total correct count differs from matrix trace")
    equal_number(metrics["overall_accuracy"], correct_count / 10000, "overall_accuracy")
    per_class = metrics["per_class"]
    require(len(per_class) == 10, "Expected exactly 10 per-class metric records")
    class_records = {}
    for record in per_class:
        class_id = record["class_id"]
        require(isinstance(class_id, int) and 0 <= class_id < 10 and class_id not in class_records,
                "Invalid/duplicate per-class identifier")
        class_records[class_id] = record
    macro_f1 = 0.0
    for index in range(10):
        record = class_records[index]
        true_positive = matrix[index][index]
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / 1000
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        require(record["class_name"] == CLASSES[index], "Per-class name differs")
        require(record["support"] == 1000 and record["correct"] == true_positive,
                "Per-class support/correct count differs")
        for key, expected in (("accuracy", recall), ("recall", recall), ("precision", precision), ("f1", f1)):
            equal_number(record[key], expected, "{} {}".format(CLASSES[index], key))
        macro_f1 += f1 / 10
    equal_number(metrics["macro_f1"], macro_f1, "macro_f1")
    # The error examples must agree with the documented, deterministic selection rule.
    expected_errors = sorted(wrong_rows, key=lambda row: (-row[3], row[0]))[:12]
    stored_errors = metrics.get("top_confident_errors", [])
    require(len(stored_errors) == len(expected_errors), "Error-example count differs")
    for stored, (index, label, prediction, confidence) in zip(stored_errors, expected_errors):
        require(stored["test_index"] == index and stored["true_class"] == CLASSES[label]
                and stored["predicted_class"] == CLASSES[prediction], "Selected error example differs")
        equal_number(stored["confidence"], confidence, "Error-example confidence")
    return {"test_samples": 10000, "correct": correct_count,
            "overall_accuracy": correct_count / 10000, "macro_f1": macro_f1,
            "test_true_labels_sha256": hashlib.sha256(true_labels).hexdigest(),
            "per_class_accuracy": {CLASSES[index]: matrix[index][index] / 1000 for index in range(10)},
            "confusion_matrix_orientation": "rows=true class, columns=predicted class"}


def check_history(directory, summary, metrics):
    with (directory / "history.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), "Training history is empty")
    epochs = [int(row["epoch"]) for row in rows]
    require(epochs == list(range(1, len(rows) + 1)), "History epochs must be consecutive starting at 1")
    for row in rows:
        for key in ("train_accuracy", "val_accuracy"):
            value = finite_number(float(row[key]), key)
            require(0 <= value <= 1, "Accuracy is outside [0, 1]")
        for key in ("train_loss", "val_loss", "seconds", "lr"):
            value = finite_number(float(row[key]), key)
            require(value >= 0, "{} is negative".format(key))
    best_row = max(rows, key=lambda row: float(row["val_accuracy"]))
    best_epoch = int(best_row["epoch"])
    best_accuracy = float(best_row["val_accuracy"])
    require(summary["completed_epochs"] == len(rows), "Summary completed_epochs differs from history")
    require(summary["best_epoch"] == best_epoch, "Summary best_epoch is not the earliest validation maximum")
    require(metrics["selected_epoch"] == best_epoch, "Evaluated checkpoint was not the validation-selected epoch")
    equal_number(summary["best_val_accuracy"], best_accuracy, "summary.best_val_accuracy")
    equal_number(metrics["best_val_accuracy"], best_accuracy, "test_metrics.best_val_accuracy")
    config = summary["config"]
    require(len(rows) <= config["epochs"], "History exceeds the configured epoch budget")
    require(len(rows) == config["epochs"] or summary["stopped_early"] is True,
            "Incomplete training was not marked as an early stop")
    seconds = finite_number(summary["training_seconds"], "training_seconds")
    require(seconds + 1 >= sum(float(row["seconds"]) for row in rows),
            "Total training time is shorter than summed epoch times")
    return {"completed_epochs": len(rows), "best_epoch": best_epoch,
            "best_val_accuracy": best_accuracy, "training_seconds": seconds}


def check_split(directory, summary):
    split = read_json(directory / "split_indices.json")
    training, validation = split["train_indices"], split["val_indices"]
    for name, indices, expected in (("train", training, 45000), ("val", validation, 5000)):
        require(len(indices) == expected, "{} size is not {}".format(name, expected))
        require(all(isinstance(index, int) and not isinstance(index, bool) for index in indices),
                "Split indices must be integers")
        require(len(set(indices)) == expected, "Duplicate {} indices".format(name))
        require(split[name + "_size"] == expected, "Stored {} split size differs".format(name))
        require(index_sha256(indices) == split[name + "_sha256"], "{} split SHA256 differs".format(name))
        require(split[name + "_class_counts"] == [expected // 10] * 10,
                "Stored {} class counts are not balanced".format(name))
    require(not set(training).intersection(validation), "Training and validation indices overlap")
    require(set(training).union(validation) == set(range(50000)), "Splits do not partition the 50000 training images")
    require(split["test_used_for_training"] is False, "Split metadata reports test-set training")
    for key, value in split.items():
        if not key.endswith("_indices"):
            require(summary["split"][key] == value, "Split summary differs for {}".format(key))
    config = summary["config"]
    require(config["val_size"] == 5000 and config["train_limit"] == 0 and config["val_limit"] == 0,
            "Result was not trained on the full 45000/5000 partition")
    return {"train_size": 45000, "val_size": 5000,
            "train_sha256": split["train_sha256"], "val_sha256": split["val_sha256"],
            "class_count_check": "Stored counts agree across split/summary: train=4500/class, val=500/class; raw labels not reloaded"}


def check_checkpoint(directory, metrics, summary, split):
    expected_hash = metrics["checkpoint_sha256"]
    require(isinstance(expected_hash, str) and len(expected_hash) == 64
            and all(character in "0123456789abcdef" for character in expected_hash),
            "Invalid checkpoint SHA256 record")
    checkpoint_path = directory / "best.pt"
    if not checkpoint_path.is_file():
        return {"status": "not_present", "recorded_sha256": expected_hash,
                "note": "Checkpoint is optional in the shared repository; hash/metadata cannot be rechecked without it"}
    actual_hash = sha256(checkpoint_path)
    require(actual_hash == expected_hash, "best.pt SHA256 differs from the evaluated checkpoint")
    if importlib.util.find_spec("torch") is None:
        return {"status": "sha256_verified_metadata_skipped", "sha256": actual_hash,
                "note": "PyTorch is not installed; checkpoint contents were not loaded"}
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["epoch"] == summary["best_epoch"], "Local checkpoint epoch differs")
    require(checkpoint["model_name"] == summary["model_name"] == metrics["model_name"],
            "Local checkpoint model name differs")
    equal_number(checkpoint["best_val_accuracy"], summary["best_val_accuracy"], "checkpoint.best_val_accuracy")
    require(checkpoint["best_epoch"] == summary["best_epoch"], "Local checkpoint best_epoch differs")
    require(checkpoint["split_hashes"] == [split["train_sha256"], split["val_sha256"]],
            "Local checkpoint split hashes differ")
    # Path/continuation fields can legitimately change if a resumed run moved files.
    ignored_config_fields = {"data_dir", "output_dir", "resume"}
    checkpoint_config = {key: value for key, value in checkpoint["config"].items()
                         if key not in ignored_config_fields}
    summary_config = {key: value for key, value in summary["config"].items()
                      if key not in ignored_config_fields}
    require(checkpoint_config == summary_config, "Local checkpoint training configuration differs")
    return {"status": "sha256_and_metadata_verified", "sha256": actual_hash,
            "epoch": checkpoint["epoch"], "model_name": checkpoint["model_name"]}


def source_manifest():
    paths = list((REPOSITORY / "src").rglob("*.py"))
    paths += list((REPOSITORY / "tools").rglob("*.py"))
    paths += list((REPOSITORY / "tests").rglob("*.py"))
    paths += [REPOSITORY / name for name in ("README.md", "requirements.txt", "run_experiments.ps1", ".gitignore")]
    return {path.relative_to(REPOSITORY).as_posix(): sha256(path)
            for path in sorted(set(paths)) if path.is_file()}


def verify(results_directory):
    verified = {}
    shared_split = None
    shared_test_labels = None
    for name, (model_name, augmentation) in GROUPS.items():
        directory = results_directory / name
        summary = read_json(directory / "summary.json")
        metrics = read_json(directory / "test_metrics.json")
        require(read_json(directory / "config.json") == summary["config"],
                "config.json differs from the summary configuration for {}".format(name))
        require(summary["model_name"] == metrics["model_name"] == model_name,
                "Unexpected model for {}".format(name))
        require(summary["config"]["model"] == model_name and summary["config"]["augment"] == augmentation,
                "Unexpected architecture/augmentation configuration for {}".format(name))
        prediction_results = check_predictions(directory, metrics)
        if shared_test_labels is None:
            shared_test_labels = prediction_results["test_true_labels_sha256"]
        require(prediction_results["test_true_labels_sha256"] == shared_test_labels,
                "Experiment groups have different true labels for the same test indices")
        history_results = check_history(directory, summary, metrics)
        split_results = check_split(directory, summary)
        current_split = (split_results["train_sha256"], split_results["val_sha256"])
        if shared_split is None:
            shared_split = current_split
        require(current_split == shared_split, "Experiment groups use different training/validation partitions")
        checkpoint_results = check_checkpoint(directory, metrics, summary, split_results)
        evidence_files = ("predictions.csv", "test_metrics.json", "history.csv", "summary.json", "split_indices.json", "config.json")
        verified[name] = {
            "passed": True, "model_name": model_name, "augmentation": augmentation,
            **prediction_results, **history_results, "split": split_results,
            "checkpoint": checkpoint_results,
            "evidence_sha256": {filename: sha256(directory / filename)
                                for filename in evidence_files if (directory / filename).is_file()},
        }
    return {
        "passed": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_directory": str(results_directory.resolve()),
        "verification_method": "Independent standard-library reconstruction from 10000 prediction rows/group; complete train/validation partition and checkpoint evidence cross-checks",
        "scope": "Consistency verification of saved evidence; does not rerun inference or independently reload CIFAR labels",
        "same_train_validation_partition": True,
        "same_test_true_labels": True,
        "source_sha256": source_manifest(), "experiments": verified,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY / "results")
    args = parser.parse_args()
    output = args.results_dir / "verification.json"
    # A failed rerun must not leave an earlier passing record looking current.
    if output.exists():
        output.unlink()
    result = verify(args.results_dir)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"passed": True, "experiments": len(result["experiments"]),
                      "output": str(output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
