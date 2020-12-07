from typing import List

import numpy as np
import tensorflow.keras.backend as K

from ucca4bpm.util.history import Epoch


class MaskedAccuracy:
    def __init__(self, mask_input, name=None):
        self._mask = mask_input
        if name is None:
            self.__name__ = MaskedAccuracy.__name__
        else:
            self.__name__ = name

    def __call__(self, y_true, y_pred):
        # vector with True for correct predictions
        equals = K.equal(K.argmax(y_pred, axis=-1),
                         K.argmax(y_true, axis=-1))

        # flatten mask, which comes in batches
        self._mask = K.flatten(self._mask)
        # set predictions for ignored class to 0
        equals = equals * self._mask

        # sum up correct predictions (ignored classes are zero, see step above)
        masked_accuracy = K.sum(equals)

        # weigh by non-ignored classes, that have to be predicted
        masked_accuracy = masked_accuracy / K.sum(self._mask)

        return masked_accuracy


def chunk_indices_by_change(classes: np.ndarray, none_class_id=0) -> List[List[int]]:
    ret = []
    last_class = None
    for i, class_id in enumerate(classes):
        if last_class != class_id:
            if class_id != none_class_id:
                ret.append([])
            last_class = class_id
        if class_id != none_class_id:
            ret[-1].append(i)
    return ret


def any_chunk_ends_with(chunks: List[List[int]], what: int):
    return any([c[-1] == what for c in chunks])


def any_chunk_starts_with(chunks: List[List[int]], what: int):
    return any([c[0] == what for c in chunks])


def span_matcher(y_true: np.ndarray, y_pred: np.ndarray, none_class_id=0, mode='exact'):
    assert mode in ['exact', 'left', 'right', 'left/right', 'fragment', 'partial']

    gold_spans = chunk_indices_by_change(y_true, none_class_id)
    predicted_spans = chunk_indices_by_change(y_pred, none_class_id)

    num_gold = len(gold_spans)
    num_pred = len(predicted_spans)
    num_ok = 0

    for indices in gold_spans:
        if mode == 'exact':
            if np.array_equal(y_true[indices], y_pred[indices]):
                if any_chunk_ends_with(predicted_spans, indices[-1]):
                    if any_chunk_starts_with(predicted_spans, indices[0]):
                        num_ok += 1
        elif mode == 'right':
            if y_true[indices[-1]] == y_pred[indices[-1]] and any_chunk_ends_with(predicted_spans, indices[-1]):
                num_ok += 1
        elif mode == 'left':
            if y_true[indices[0]] == y_pred[indices[0]] and any_chunk_starts_with(predicted_spans, indices[0]):
                num_ok += 1
        elif mode == 'left/right':
            if y_true[indices[-1]] == y_pred[indices[-1]] and any_chunk_ends_with(predicted_spans, indices[-1]):
                num_ok += 1
            elif y_true[indices[0]] == y_pred[indices[0]] and any_chunk_starts_with(predicted_spans, indices[0]):
                num_ok += 1
        elif mode == 'fragment':
            split_gold_spans = [i for s in gold_spans for i in s]
            split_predicted_spans = [i for s in predicted_spans for i in s]
            num_gold = len(split_gold_spans)
            num_pred = len(split_predicted_spans)
            num_ok = int(np.sum(y_true[split_gold_spans] == y_pred[split_gold_spans]))
        elif mode == 'partial':
            span_boundaries_match = any_chunk_ends_with(predicted_spans, indices[-1]) \
                                    and any_chunk_starts_with(predicted_spans, indices[0])
            if np.all(y_true[indices] == y_pred[indices]) and span_boundaries_match:
                # exact match
                num_ok += 1
            elif np.any(y_true[indices] == y_pred[indices]):
                # partial match, either because boundaries did not match
                # and/or because some value was not predicted properly
                num_ok += .5

    return num_gold, num_pred, num_ok


def get_metrics(epoch: Epoch, span_matching_modes: List[str]):
    metrics = {}
    for span_matching_mode in span_matching_modes:
        spans = [float(sum(x)) for x in zip(*epoch.metrics[f'span_{span_matching_mode}'])]
        num_gold, num_pred, num_ok = spans
        recall = (num_ok / num_gold) if num_gold != 0 else 0
        precision = (num_ok / num_pred) if num_pred != 0 else 0
        f1 = 0
        if recall + precision != 0:
            f1 = 2 * recall * precision / (recall + precision)

        metrics.update({
            f'recall_{span_matching_mode}': recall,
            f'precision_{span_matching_mode}': precision,
            f'f1_{span_matching_mode}': f1
        })
    return metrics
