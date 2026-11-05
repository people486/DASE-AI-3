"""Checks for data leakage, metric semantics, and usable model gradients.

Run from the repository root: python -m unittest discover -s tests -v
No dataset download or full training is required.
"""

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.evaluate import gradcam, summarize_predictions
from src.models import build_model
from src.train import build_loaders, stratified_take


class SplitTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_complete_and_balanced(self):
        # Mimic the official train-set class distribution without image files.
        labels = np.repeat(np.arange(10), 5000)
        original = list(range(50000))
        validation, training = stratified_take(original, labels, 5000, 42)
        self.assertEqual(len(validation), 5000)
        self.assertEqual(len(training), 45000)
        self.assertFalse(set(validation) & set(training))
        self.assertEqual(set(validation) | set(training), set(original))
        np.testing.assert_array_equal(np.bincount(labels[validation]), [500] * 10)
        np.testing.assert_array_equal(np.bincount(labels[training]), [4500] * 10)
        self.assertEqual((validation, training), stratified_take(original, labels, 5000, 42))
        self.assertNotEqual(validation, stratified_take(original, labels, 5000, 43)[0])

    def test_validation_uses_independent_nonrandom_transform(self):
        calls = []

        class FakeCIFAR10:
            def __init__(self, root, train, download, transform):
                calls.append(train)
                self.targets = np.repeat(np.arange(10), 10).tolist()
                self.transform = transform

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, index):
                pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
                return self.transform(Image.fromarray(pixels)), self.targets[index]

        args = argparse.Namespace(
            augment="standard", data_dir=Path("unused"), val_size=20,
            seed=42, train_limit=0, val_limit=0, batch_size=8, num_workers=0,
        )
        with patch("src.train.datasets.CIFAR10", FakeCIFAR10):
            training, validation, _, split = build_loaders(args, torch.device("cpu"))
        self.assertEqual(calls, [True, True], "Training must never load the test split")
        self.assertIsNot(training.dataset.dataset, validation.dataset.dataset)
        train_ops = training.dataset.dataset.transform.transforms
        val_ops = validation.dataset.dataset.transform.transforms
        self.assertTrue(any(isinstance(op, transforms.RandomCrop) for op in train_ops))
        self.assertTrue(any(isinstance(op, transforms.RandomHorizontalFlip) for op in train_ops))
        self.assertFalse(any(isinstance(op, (transforms.RandomCrop, transforms.RandomHorizontalFlip)) for op in val_ops))
        self.assertFalse(set(split["train_indices"]) & set(split["val_indices"]))
        first, second = validation.dataset[0][0], validation.dataset[0][0]
        self.assertTrue(torch.equal(first, second))


class MetricTests(unittest.TestCase):
    def test_confusion_orientation_and_precision_recall(self):
        metrics = summarize_predictions(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        matrix = np.asarray(metrics["confusion_matrix"])
        np.testing.assert_array_equal(matrix[:2, :2], [[1, 1], [0, 2]])
        self.assertEqual(int(matrix.sum()), 4)
        self.assertAlmostEqual(metrics["overall_accuracy"], 0.75)
        first, second = metrics["per_class"][:2]
        self.assertEqual(first["support"], 2)
        self.assertAlmostEqual(first["accuracy"], 0.5)
        self.assertAlmostEqual(first["recall"], 0.5)
        self.assertAlmostEqual(first["precision"], 1.0)
        self.assertAlmostEqual(second["recall"], 1.0)
        self.assertAlmostEqual(second["precision"], 2 / 3)
        self.assertAlmostEqual(second["f1"], 0.8)

    def test_invalid_metric_inputs_are_rejected(self):
        examples = (
            (np.array([], dtype=int), np.array([], dtype=int)),
            (np.array([0, 1]), np.array([0])),
            (np.array([10]), np.array([0])),
            (np.array([-1]), np.array([0])),
            (np.array([0.5]), np.array([0])),
        )
        for labels, predictions in examples:
            with self.subTest(labels=labels, predictions=predictions):
                with self.assertRaises(ValueError):
                    summarize_predictions(labels, predictions)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_models_receive_finite_training_gradients(self):
        for name in ("baseline", "improved"):
            with self.subTest(model=name):
                torch.manual_seed(7)
                model = build_model(name)
                logits = model(torch.randn(2, 3, 32, 32))
                self.assertEqual(tuple(logits.shape), (2, 10))
                nn.CrossEntropyLoss()(logits, torch.tensor([1, 8])).backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
                self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
                self.assertTrue(any(value.abs().sum().item() > 0 for value in gradients))

    def test_gradcam_is_finite_spatial_and_preserves_weights(self):
        for name in ("baseline", "improved"):
            with self.subTest(model=name):
                torch.manual_seed(7)
                model = build_model(name).eval()
                before = {key: value.clone() for key, value in model.state_dict().items()}
                heatmap, positive = gradcam(model, torch.randn(3, 32, 32), target=1)
                self.assertEqual(heatmap.shape, (32, 32))
                self.assertTrue(np.isfinite(heatmap).all())
                self.assertGreaterEqual(float(heatmap.min()), 0)
                self.assertLessEqual(float(heatmap.max()), 1.00001)
                self.assertIsInstance(positive, bool)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(before[key], value), key)


if __name__ == "__main__":
    unittest.main()
