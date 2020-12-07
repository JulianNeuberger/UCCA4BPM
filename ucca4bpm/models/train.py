import datetime
import json
import os
import pickle
import random
import time
from collections import defaultdict
from typing import List, Optional, Dict

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.python.keras.losses import Loss

from ucca4bpm.models.conv import build_model
from ucca4bpm.util.history import History, Epoch, Run
from ucca4bpm.util.metrics import span_matcher, get_metrics


def report_samples_stats(samples, class_names: List[str], message=''):
    targets = []
    for g, _ in samples:
        target_mask = tf.squeeze(tf.where(g.ndata['is_target']))
        target_classes = tf.gather(g.ndata['class_ordinal'], target_mask).numpy()

        # convert to an array of unique classes contained in this
        target_classes = np.unique(target_classes)
        target_classes = np.atleast_1d(target_classes)
        targets.append(target_classes)
    targets = np.concatenate(targets)

    classes, class_occurrences = np.unique(targets, return_counts=True)
    classes_formatted = ', '.join([f'{class_names[c]}: {o}' for c, o in zip(classes, class_occurrences)])

    print(f'{message}{classes_formatted}.')


def print_progress(epoch: Epoch, num_total_steps):
    mean_duration_per_sample = epoch.mean_iteration_duration()
    samples_remaining = num_total_steps - epoch.samples_seen()

    eta = mean_duration_per_sample * samples_remaining

    progress = f'Done with {epoch.samples_seen():{len(str(num_total_steps))}d} of {num_total_steps} batches'
    metrics = f'Loss: {epoch.loss():.5f}, Acc: {epoch.accuracy():3.2%}'
    eta_f = f'ETA: {datetime.timedelta(seconds=int(eta))}'

    print(f'\r{progress} | {eta_f} | {metrics}', end='')


def run_on_samples(model, optimizer, loss_fn, acc_fn, samples, max_num_nodes, classes, edge_classes,
                   verbosity=0, class_weights: Optional[np.ndarray] = None,
                   span_matching_modes=None, explain=False, none_class_id=None,
                   training=False, use_manual_weight_decay=True) -> Epoch:
    if span_matching_modes is None:
        span_matching_modes = ['exact']
    epoch = Epoch()
    metrics = defaultdict(list)
    for i, (sample, texts) in enumerate(samples):
        start = time.time()
        with tf.device("/cpu:0"):
            with tf.GradientTape() as tape:
                g: dgl.DGLHeteroGraph = sample

                # load inputs and ensure correct types
                # the DGL library is very specific about types (int32 vs int64)
                target_classes = g.ndata['class_one_hot']
                target_classes = tf.cast(target_classes, tf.float32)
                node_features = g.ndata['feature']
                if node_features.dtype.is_integer:
                    # this happens if we pass ordinal node ids as features (== featureless mode)
                    # dgl needs them in int64, so let's cast them here
                    node_features = tf.one_hot(node_features, depth=max_num_nodes)  # tf.cast(node_features, tf.int64)
                else:
                    # this happens in every other case, like e.g. embeddings
                    # here dgl wants float32
                    node_features = tf.cast(node_features, tf.float32)
                edge_types = g.edata['class_ordinal']
                edge_types = tf.cast(edge_types, tf.int64)

                target_mask = g.ndata['is_target']
                target_mask = tf.squeeze(tf.where(target_mask))

                g = dgl.to_homogeneous(g)

                predicted_classes = model(g, node_features, edge_types, training=training)

                predicted_classes = tf.gather(predicted_classes, target_mask)
                target_classes = tf.gather(target_classes, target_mask)

                predicted_classes_ord = np.atleast_1d(np.argmax(predicted_classes.numpy(), axis=-1))
                target_classes_ord = np.atleast_1d(np.argmax(target_classes.numpy(), axis=-1))

                loss = loss_fn(target_classes, predicted_classes)
                acc = acc_fn(target_classes, predicted_classes)

                if class_weights is not None:
                    idxs = target_classes_ord
                    class_weighting = class_weights[idxs]
                    loss = loss * class_weighting

                if training:
                    if use_manual_weight_decay:
                        # Manually Weight Decay
                        # We found Tensorflow has a different implementation on weight decay
                        # of Adam(W) optimizer with PyTorch. And this results in worse results.
                        # Manually adding weights to the loss to do weight decay solves this problem.
                        for weight in model.trainable_weights:
                            loss += 1e-4 * tf.nn.l2_loss(weight)

                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

                epoch.update_with_sample(time.time() - start, loss.numpy().mean(), acc.numpy().mean(),
                                         target_classes_ord, predicted_classes_ord)

                for span_matching_mode in span_matching_modes:
                    span_metric = span_matcher(target_classes_ord, predicted_classes_ord, mode=span_matching_mode,
                                               none_class_id=none_class_id)
                    epoch.metrics[f'span_{span_matching_mode}'].append(span_metric)

        if verbosity > 0:
            print_progress(epoch, len(samples))

    if verbosity > 0:
        # finish printing by adding new line after the carriage return (stopping continuous output)
        print()
    return epoch


def write_history(histories: List[History], target_dir, file_name):
    fold_metrics = []
    for fold_id, history in enumerate(histories):
        epoch = history.epochs[-1]
        fold_metrics.append({
            'loss': epoch.loss(),
            'y_true': [a.tolist() for a in epoch.y_true_ord],
            'y_pred': [a.tolist() for a in epoch.y_pred_ord]
        })
    with open(os.path.join(target_dir, file_name) + '.json', 'w') as f:
        json.dump(fold_metrics, f)


def fit(model, optimizer, loss_fn, accuracy_fn,
        train_samples, validate_samples, max_num_nodes,
        class_names, edge_class_names,
        span_matching_modes: List[str],
        num_epochs: int = 1, validate_every_epoch=True,
        class_weights: Optional[Dict[int, float]] = None,
        manual_weight_decay=True, none_class_id=None,
        early_stopping=False, stop_criterion_name='macro_f1', patience=2) -> History:
    assert not early_stopping or validate_every_epoch, \
        'If you want to use early stopping, we have to validate after each epoch!'

    if class_weights is not None:
        class_weights_arr = np.ones((len(class_weights),))
        for class_id, class_weight in class_weights.items():
            class_weights_arr[class_id] = class_weight
        class_weights = class_weights_arr

    history = History([])
    stop_criterion: Optional[float] = None
    best_model_weights = None
    current_patience = patience
    for epoch_id in range(1, num_epochs + 1):
        print(f'--- Epoch {epoch_id}/{num_epochs} {"-" * 50}')

        should_validate_this_epoch = validate_every_epoch or epoch_id == num_epochs

        random.shuffle(train_samples)

        print('Starting training:')
        run_on_samples(
            model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            acc_fn=accuracy_fn,
            samples=train_samples,
            class_weights=class_weights,
            classes=class_names,
            max_num_nodes=max_num_nodes,
            verbosity=1, training=True,
            none_class_id=none_class_id,
            use_manual_weight_decay=manual_weight_decay,
            span_matching_modes=span_matching_modes,
            edge_classes=edge_class_names
        )

        if should_validate_this_epoch:
            print('Starting validation:')
            validation_epoch = run_on_samples(
                model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                acc_fn=accuracy_fn,
                samples=validate_samples,
                class_weights=class_weights,
                classes=class_names,
                max_num_nodes=max_num_nodes,
                none_class_id=none_class_id,
                explain=False,
                verbosity=1, training=False,
                span_matching_modes=span_matching_modes,
                edge_classes=edge_class_names
            )
            target_classes_ord = np.concatenate(validation_epoch.y_true_ord)
            predicted_classes_ord = np.concatenate(validation_epoch.y_pred_ord)
            current_metrics = get_metrics(validation_epoch, span_matching_modes=span_matching_modes)
            validation_epoch.metrics.update(current_metrics)
            print_model_metrics(validation_epoch.metrics, keys=[f'f1_{s}' for s in span_matching_modes])
            if early_stopping:
                if stop_criterion is None or stop_criterion < current_metrics[stop_criterion_name]:
                    # criterion increased, continue training
                    best_model_weights = model.get_weights()
                    stop_criterion = current_metrics[stop_criterion_name]
                    current_patience = patience
                else:
                    # criterion is worsening, decrease patience
                    current_patience -= 1
                    if current_patience <= 0:
                        print(f'Stopping training early in epoch {epoch_id}/{num_epochs}.')
                        # stop early, reset model
                        model.set_weights(best_model_weights)
                        break
            history.epochs.append(validation_epoch)
    return history


def print_model_metrics(metrics, keys):
    formatted_metrics = [f'{key}: {metrics[key]:.2%}' for key in keys]
    print(f'{", ".join(formatted_metrics)}')


def target_combination_to_ordinal(target_classes: tf.Tensor, target_mask: tf.Tensor) -> int:
    target_mask = tf.squeeze(tf.where(target_mask))
    target_classes = tf.gather(target_classes, target_mask).numpy()

    # convert to an array of unique classes contained in this
    target_classes = np.unique(target_classes)
    target_classes = np.atleast_1d(target_classes)

    # create an array, where we treat each class as bit in a bit vector
    # e.g. class combination [2, 5] would result in vector [0, 0, 1, 0, 0, 1], or here [4, 32]
    target_classes = np.power(2, target_classes)

    # sum vector to get int for unique combinations of classes
    return np.sum(target_classes, dtype=np.int32).item()


def cross_validate(samples, k_folds: int, num_epochs: int, early_stopping: bool,
                   max_num_nodes: int, optimizer, loss, num_bases: int, dropout: float,
                   feature_len: int, hidden_len: int, target_len: int, class_names: List[str],
                   node_texts: Dict[int, Dict[int, str]], edge_classes, report_splits_to: str,
                   num_edge_classes: int, num_hidden_layers: int, class_weights: Dict[int, float],
                   none_class_id: Optional[int],
                   span_matching_modes: List[str], dry_run: bool):
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=12)
    fold_histories = []
    indices = np.array(list(samples.keys()))
    indices = indices.reshape(-1, 1)
    target_classes = [g.ndata['class_ordinal'] for g in samples.values()]
    target_masks = [g.ndata['is_target'] for g in samples.values()]
    targets = [
        target_combination_to_ordinal(cls, msk)
        for cls, msk
        in zip(target_classes, target_masks)
    ]
    targets = np.array(targets)

    splits = list(kfold.split(X=indices, y=targets))
    if report_splits_to is not None:
        with open(report_splits_to, 'w') as f:
            json.dump([(ts.tolist(), vs.tolist()) for ts, vs in splits], f)

    for fold, (train_indices, validate_indices) in enumerate(splits):
        model = build_model(features_len=feature_len, hidden_len=hidden_len, target_len=target_len,
                            num_bases=num_bases, dropout=dropout,
                            num_edge_types=num_edge_classes, num_hidden_layers=num_hidden_layers)
        if dry_run:
            return
        print(f'=== Fold {fold + 1} / {kfold.n_splits} {"=" * 50}')

        train_samples = [(samples[idx], node_texts[idx]) for idx in train_indices]
        validate_samples = [(samples[idx], node_texts[idx]) for idx in validate_indices]

        history = fit(
            model=model,
            optimizer=optimizer,
            loss_fn=loss,
            class_names=class_names,
            accuracy_fn=categorical_accuracy,
            train_samples=train_samples,
            validate_samples=validate_samples,
            class_weights=class_weights,
            max_num_nodes=max_num_nodes,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            span_matching_modes=span_matching_modes,
            edge_class_names=edge_classes,
            none_class_id=none_class_id,
        )
        fold_histories.append(history)
        print()

    return fold_histories


def run(run_name: str, experiment_name: str, data_set_path: str, none_class_id: Optional[int],
        num_epochs=1, early_stopping=False, num_bases: int = None,
        dropout=0.0, span_matching_modes: List[str] = None,
        optimizer: Optimizer = Adam(lr=0.01), loss: Loss = CategoricalCrossentropy(from_logits=False),
        gcn_hidden_len=10, num_gcn_hidden_layers=2,
        num_folds=5, dry_run=False) -> Optional[Run]:
    random.seed(1337)
    tf.random.set_seed(42)

    run_export_dir = os.path.join('ucca4bpm/runs', experiment_name)
    os.makedirs(run_export_dir, exist_ok=True)

    run_history = Run(run_name)

    print(f'### Experiment: {run_name} ################')
    with open(data_set_path, 'rb') as f:
        data = pickle.load(f)
    samples = data['dgl_graphs']
    target_classes: List[str] = data['node_classes']
    num_target_classes = len(target_classes)
    target_class_weights: dict = data['class_weights']
    edge_classes = data['edge_classes']
    num_edge_classes = len(edge_classes)
    max_num_nodes = data['max_num_nodes']
    node_feature_len = data['node_feature_len']
    node_texts = data['node_texts']
    if node_feature_len == 1:
        # we only have an id as feature, the actual input
        # length will be the one hot encoded version of that
        node_feature_len = max_num_nodes

    # estimate quality of model fit
    fold_histories = cross_validate(
        samples=samples,
        k_folds=num_folds,
        num_epochs=num_epochs,
        max_num_nodes=max_num_nodes,
        optimizer=optimizer,
        loss=loss,
        dropout=dropout,
        report_splits_to=f'{run_export_dir}/{run_name}_splits.json',
        early_stopping=early_stopping,
        feature_len=node_feature_len,
        hidden_len=gcn_hidden_len,
        target_len=num_target_classes,
        class_names=target_classes,
        class_weights=target_class_weights,
        num_hidden_layers=num_gcn_hidden_layers,
        num_bases=num_bases,
        num_edge_classes=num_edge_classes,
        node_texts=node_texts,
        span_matching_modes=span_matching_modes,
        edge_classes=edge_classes,
        none_class_id=none_class_id,
        dry_run=dry_run
    )

    if fold_histories is None and dry_run:
        return

    run_history.fold_histories = fold_histories
    write_history(fold_histories, run_export_dir, f'{run_name}')

    return run_history
