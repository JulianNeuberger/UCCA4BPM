import json
from typing import List, Optional, Iterable, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mc
import colorsys

from ucca4bpm.util.history import Run, History, Epoch
from ucca4bpm.util.metrics import span_matcher, get_metrics


def plot_f1_metrics(run_histories: List[Run],
                    experiment_name: str,
                    metric_names: Dict[str, str],
                    target_file_paths: Optional[List[str]] = None,
                    axis: plt.Axes = None,
                    y_lims: Tuple[float, float] = None,
                    mode='box',
                    add_legend=True,
                    color=None):
    standalone_mode = axis is None

    if color is None:
        # by default we use bright orange from map 'tab20c'
        color = plt.get_cmap('tab20c')(4)

    runs_data = []
    for run_history in run_histories:
        metrics_data = []
        for metric_name in metric_names.keys():
            metric_data = []
            for fold_history in run_history.fold_histories:
                # add the metric for the last epoch
                metric_data.append(fold_history.epochs[-1].metrics[metric_name])
            metric_data = np.array(metric_data)
            if mode == 'bar':
                metric_data = np.mean(metric_data)
            metrics_data.append(metric_data)
        runs_data.append(metrics_data)

    with plt.style.context('seaborn'):
        if axis is None:
            fig: plt.Figure = plt.figure()
            fig.suptitle(experiment_name, fontsize=24)
            axis = fig.add_subplot(111)
        else:
            axis.set_title(experiment_name, fontsize=22)
        axis.set_ylabel('Metric Value', fontsize=22)
        if y_lims is not None:
            print('limits', y_lims)
            axis.set_ylim(*y_lims)
        num_metrics = None
        num_runs = len(runs_data)
        for i, metrics_data in enumerate(runs_data):
            num_metrics = len(metrics_data)
            xs = range(i * len(metrics_data) + i, (i + 1) * len(metrics_data) + i)

            max_v = .9
            min_v = .6
            colors = []
            for idx in range(num_metrics):
                if num_metrics > 1:
                    norm = idx * (max_v - min_v) / (num_metrics - 1)
                else:
                    norm = 0
                fill_color = list(colorsys.rgb_to_hls(*mc.to_rgb(color)))
                fill_color[1] = min_v + norm
                colors.append((
                    color,
                    colorsys.hls_to_rgb(*fill_color)
                ))
            line_styles = ['-', '-.', ':', '--']

            if mode == 'box':
                boxplots = axis.boxplot(
                    metrics_data,
                    meanline=True,
                    showmeans=True,
                    positions=xs,
                    widths=0.6,
                    patch_artist=True
                )

                for plot_idx in range(num_metrics):
                    dark_color = colors[plot_idx][0]
                    light_color = colors[plot_idx][1]

                    plt.setp(boxplots['boxes'][plot_idx], color=dark_color)
                    plt.setp(boxplots['boxes'][plot_idx], facecolor=light_color)
                    plt.setp(boxplots['boxes'][plot_idx], linestyle=line_styles[plot_idx])

                    plt.setp(boxplots['whiskers'][plot_idx * 2], color=dark_color)
                    plt.setp(boxplots['whiskers'][plot_idx * 2 + 1], color=dark_color)
                    plt.setp(boxplots['whiskers'][plot_idx * 2], linestyle=line_styles[plot_idx])
                    plt.setp(boxplots['whiskers'][plot_idx * 2 + 1], linestyle=line_styles[plot_idx])

                    plt.setp(boxplots['caps'][plot_idx * 2], color=dark_color)
                    plt.setp(boxplots['caps'][plot_idx * 2 + 1], color=dark_color)

                    plt.setp(boxplots['fliers'][plot_idx], markeredgecolor=dark_color)
                    plt.setp(boxplots['fliers'][plot_idx], marker='x')

                    plt.setp(boxplots['medians'][plot_idx], color=dark_color)
                    plt.setp(boxplots['means'][plot_idx], color=dark_color)

                    legend_styles = [boxplots['boxes'][idx] for idx in range(num_metrics)]
            elif mode == 'bar':
                legend_styles = []
                for plot_idx in range(num_metrics):
                    ret = axis.bar(xs[plot_idx], metrics_data[plot_idx],
                                   color=colors[plot_idx][1],
                                   edgecolor=colors[plot_idx][0],
                                   width=0.6,
                                   linewidth=1.25,
                                   linestyle=line_styles[plot_idx], )
                    legend_styles.append(ret)

        tick_offset = num_metrics * 0.5 - 0.5
        ticks = np.arange(start=tick_offset, stop=num_runs * num_metrics + num_runs + tick_offset,
                          step=num_metrics + 1.0)
        axis.set_xticks(ticks)
        for yticklabel in axis.get_yticklabels():
            yticklabel.set_fontsize(20)
        axis.set_xticklabels([r.name for r in run_histories], fontsize=20, rotation=0)
        if add_legend:
            axis.legend(legend_styles, metric_names.values(),
                        loc='lower right', fontsize=16,
                        facecolor="white", frameon=True,
                        edgecolor="black")

        if standalone_mode:
            fig.show()
            if target_file_paths is not None:
                for target_file_path in target_file_paths:
                    fig.savefig(target_file_path)
        return legend_styles


def plot_run(run: Run, metric_names: List[str], target_file_path):
    axes: Iterable[plt.Axes]
    fig: plt.Figure
    fig, axes = plt.subplots(len(run.fold_histories))
    for history, axis in zip(run.fold_histories, axes):
        xs = np.arange(len(history.epochs))

        loss = np.array([np.concatenate(epoch.loss).mean() for epoch in history.epochs])
        axis.plot(xs, loss)

        percentage_axis: plt.Axes = axis.twinx()
        percentage_axis.set_ylim(0.0, 1.0)
        metrics_to_plot = [[] for _ in metric_names]
        for epoch in history.epochs:
            for i, metric_name in enumerate(metric_names):
                metrics_to_plot[i].append(epoch.metrics[metric_name])
        for m in metrics_to_plot:
            percentage_axis.plot(xs, np.array(m), linestyle='dashed')
    fig.show()
    fig.savefig(target_file_path)


def find_y_max_in_runs(parallel_run_histories: List[List[Run]], metric_names: List[Dict[str, str]]):
    y_max = 0
    for i, run_histories in enumerate(parallel_run_histories):
        for run_history in run_histories:
            for fold_history in run_history.fold_histories:
                epoch = fold_history.epochs[-1]
                for metric_name in metric_names[i].keys():
                    if epoch.metrics[metric_name] > y_max:
                        y_max = epoch.metrics[metric_name]
    return y_max


def find_y_min_in_runs(parallel_run_histories: List[List[Run]], metric_names: List[Dict[str, str]]):
    y_min = 0
    for i, run_histories in enumerate(parallel_run_histories):
        for run_history in run_histories:
            for fold_history in run_history.fold_histories:
                epoch = fold_history.epochs[-1]
                for metric_name in metric_names[i].keys():
                    if epoch.metrics[metric_name] < y_min:
                        y_min = epoch.metrics[metric_name]
    return y_min


def plot_f1_metrics_parallel(parallel_run_histories: List[List[Run]],
                             title: str,
                             titles: List[str],
                             metric_names: List[Dict[str, str]],
                             layout=None,
                             sync_scales=False,
                             margin_bot=0.0,
                             mode='box',
                             target_file_paths: Optional[List[str]] = None,
                             color=None):
    if layout is None:
        layout = 1, len(parallel_run_histories)
    fig: plt.Figure = plt.figure(figsize=[8.0 * layout[1], 6.0 * layout[0]])
    fig.suptitle(title, fontsize=24)
    y_lims = None
    if sync_scales:
        y_max_lim = find_y_max_in_runs(parallel_run_histories, metric_names)
        y_min_lim = find_y_min_in_runs(parallel_run_histories, metric_names)

        lim_range = y_max_lim - y_min_lim
        margin = lim_range * .1

        y_min_lim = max((0, y_min_lim - margin - margin_bot))
        y_max_lim = min((1, y_max_lim + margin))

        y_lims = (y_min_lim, y_max_lim)

    with plt.style.context('seaborn-whitegrid'):
        for i, run_history in enumerate(parallel_run_histories):
            axis: plt.Axes
            axis = fig.add_subplot(layout[0], layout[1], i + 1)
            legend_styles = plot_f1_metrics(run_history, titles[i], metric_names[i], target_file_paths=None, axis=axis,
                                            color=color, mode=mode, y_lims=y_lims, add_legend=False)
            if i == len(parallel_run_histories) - 1:
                axis.legend(legend_styles, metric_names[-1].values(),
                            loc='lower right', fontsize=16,
                            facecolor="white", frameon=True,
                            edgecolor="black", bbox_to_anchor=(1.25, 0))
        fig.show()
        if target_file_paths is not None:
            for target_file_path in target_file_paths:
                fig.savefig(target_file_path)


def build_run_histories_from_paths(run_dict, matching_modes, none_class_id):
    runs = []
    for run_name, run_path in run_dict.items():
        runs.append(build_run_history_from_path(run_path, run_name, matching_modes, none_class_id))
    return runs


def build_run_history_from_path(run_path, run_name, matching_modes, none_class_id):
    with open(run_path) as run_file:
        folds = json.load(run_file)
    run_history = Run(name=run_name)
    run_history.fold_histories = []
    for epoch_vals in folds:
        epoch = Epoch()
        epoch.metrics = {
            f'span_{mode}': []
            for mode in matching_modes
        }
        y_trues = epoch_vals['y_true']
        y_preds = epoch_vals['y_pred']
        for mode in matching_modes:
            for y_true, y_pred in zip(y_trues, y_preds):
                span = span_matcher(np.array(y_true), np.array(y_pred), mode=mode, none_class_id=none_class_id)
                epoch.metrics[f'span_{mode}'].append(span)
        epoch.metrics = get_metrics(epoch, matching_modes)

        run_history.fold_histories.append(History([epoch]))
    return run_history
