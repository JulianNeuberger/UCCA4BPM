from collections import defaultdict
from typing import Dict, List, Any

import numpy as np


class Epoch:
    def __init__(self):
        self.duration = 0
        self.metrics: Dict[str, List[Any]] = defaultdict(list)

        self._samples_seen = 0
        self._loss = 0
        self._acc = 0

        self.y_true_ord = []
        self.y_pred_ord = []

    def update_with_sample(self, duration: float, loss: float, accuracy: float, y_true_ord, y_pred_ord):
        self.duration += duration

        self._loss *= self._samples_seen
        self._acc *= self._samples_seen
        self._samples_seen += 1
        self._loss += loss
        self._acc += accuracy
        self._loss /= self._samples_seen
        self._acc /= self._samples_seen

        self.y_pred_ord.append(y_pred_ord)
        self.y_true_ord.append(y_true_ord)

    def loss(self):
        return self._loss

    def accuracy(self):
        return self._acc

    def mean_iteration_duration(self):
        return self.duration / self._samples_seen

    def samples_seen(self):
        return self._samples_seen


class History:
    def __init__(self, epochs: List[Epoch]):
        self.epochs = epochs


class Run:
    def __init__(self, name: str):
        self.name = name
        self.fold_histories: List[History] = []
