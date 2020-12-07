from ucca4bpm.util.visualization import build_run_histories_from_paths, plot_f1_metrics_parallel


def plot_experiments_features():
    experiment_name = 'our_approach_our_data_features'
    parallel_run_paths = [
        {
            'No Features': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_featuresfeatureless.json',
            'POS': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_featurespostags_fine.json',
            'Glove': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_featuresglove25.json',
            'Word2Vec': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_featuresgoogle300.json',
        }
        for dataset in ['sc', 'ssr', 'srl', 'quishpi']
    ]

    none_class_ids = [None, None, None, None]

    matching_modes = ['exact', 'partial', 'fragment']
    parallel_runs = [build_run_histories_from_paths(run_dict, matching_modes, none_class_ids[i]) for i, run_dict in
                     enumerate(parallel_run_paths)]
    plot_f1_metrics_parallel(parallel_runs,
                             layout=(2, 2),
                             margin_bot=.2,
                             title='Importance of Node-Features',
                             titles=[
                                 'a) Clause Classification',
                                 'b) Clause Semantic Relation',
                                 'c) Semantic Role Labeling (MGTC)',
                                 'd) Semantic Role Labeling (ATDP)'
                             ],
                             metric_names=[
                                 {'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_exact': 'F1 (Exact)', 'f1_partial': 'F1 (Partial)',
                                  'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_exact': 'F1 (Exact)', 'f1_partial': 'F1 (Partial)',
                                  'f1_fragment': 'F1 (Fragment)'},
                             ],
                             target_file_paths=[
                                 f'ucca4bpm/figures/experiments_features.png',
                                 f'ucca4bpm/figures/experiments_features.pdf'
                             ]
                             )


def plot_experiments_lr():
    experiment_name = 'our_approach_our_data_lr'
    parallel_run_paths = [
        {
            '5e-3': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_lr_0-005.json',
            '5e-4': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_lr_0-0005.json',
            '5e-5': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_lr_5e-05.json',
            '5e-6': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_lr_5e-06.json',
        }
        for dataset in ['sc', 'ssr', 'srl', 'quishpi']
    ]

    none_class_ids = [None, None, None, None]

    matching_modes = ['exact', 'partial', 'fragment']
    parallel_runs = [build_run_histories_from_paths(run_dict, matching_modes, none_class_ids[i]) for i, run_dict in
                     enumerate(parallel_run_paths)]
    plot_f1_metrics_parallel(parallel_runs,
                             layout=(2, 2),
                             margin_bot=.2,
                             title='Importance of Learning Rate',
                             titles=[
                                 'a) Clause Classification',
                                 'b) Clause Semantic Relation',
                                 'c) Semantic Role Labeling (MGTC)',
                                 'd) Semantic Role Labeling (ATDP)'
                             ],
                             metric_names=[
                                 {'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_exact': 'F1 (Exact)', 'f1_partial': 'F1 (Partial)',
                                  'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_exact': 'F1 (Exact)', 'f1_partial': 'F1 (Partial)',
                                  'f1_fragment': 'F1 (Fragment)'},
                             ],
                             target_file_paths=[
                                 f'ucca4bpm/figures/experiments_lr.png',
                                 f'ucca4bpm/figures/experiments_lr.pdf'
                             ]
                             )


def plot_experiments_hidden_units():
    experiment_name = 'our_approach_our_data_hidden_len'
    parallel_run_paths = [
        {
            '16': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_hidden_len_16.json',
            '32': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_hidden_len_32.json',
            '64': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_hidden_len_64.json',
            '128': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_hidden_len_128.json',
        }
        for dataset in ['sc', 'ssr', 'srl', 'quishpi']
    ]

    none_class_ids = [None, None, None, None]

    matching_modes = ['exact', 'partial', 'fragment']
    parallel_runs = [build_run_histories_from_paths(run_dict, matching_modes, none_class_ids[i]) for i, run_dict in
                     enumerate(parallel_run_paths)]
    plot_f1_metrics_parallel(parallel_runs,
                             layout=(2, 2),
                             margin_bot=.2,
                             title='Importance of Hidden Units',
                             titles=[
                                 'a) Clause Classification',
                                 'b) Clause Semantic Relation',
                                 'c) Semantic Role Labeling (MGTC)',
                                 'd) Semantic Role Labeling (ATDP)'
                             ],
                             metric_names=[
                                 {'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_exact': 'F1 (Exact)', 'f1_partial': 'F1 (Partial)',
                                  'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_exact': 'F1 (Exact)', 'f1_partial': 'F1 (Partial)',
                                  'f1_fragment': 'F1 (Fragment)'},
                             ],
                             target_file_paths=[
                                 f'ucca4bpm/figures/experiments_hidden_len.png',
                                 f'ucca4bpm/figures/experiments_hidden_len.pdf'
                             ]
                             )


def plot_experiments_hidden_layers():
    experiment_name = 'our_approach_our_data_hidden_layers'
    parallel_run_paths = [
        {
            '0': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_hidden_layers_0.json',
            '2': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_hidden_layers_2.json',
            '4': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_hidden_layers_4.json',
            '8': f'ucca4bpm/runs/{experiment_name}/our_approach_our_data_{dataset}_hidden_layers_8.json',
        }
        for dataset in ['sc', 'ssr', 'srl', 'quishpi']
    ]

    none_class_ids = [None, None, None, None]

    matching_modes = ['exact', 'partial', 'fragment']
    parallel_runs = [build_run_histories_from_paths(run_dict, matching_modes, none_class_ids[i]) for i, run_dict in
                     enumerate(parallel_run_paths)]
    plot_f1_metrics_parallel(parallel_runs,
                             layout=(2, 2),
                             margin_bot=.4,
                             title='Importance of Hidden Layers',
                             titles=[
                                 'a) Clause Classification',
                                 'b) Clause Semantic Relation',
                                 'c) Semantic Role Labeling (MGTC)',
                                 'd) Semantic Role Labeling (ATDP)'
                             ],
                             metric_names=[
                                 {'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_exact': 'F1 (Exact)', 'f1_partial': 'F1 (Partial)',
                                  'f1_fragment': 'F1 (Fragment)'},
                                 {'f1_exact': 'F1 (Exact)', 'f1_partial': 'F1 (Partial)',
                                  'f1_fragment': 'F1 (Fragment)'},
                             ],
                             target_file_paths=[
                                 f'ucca4bpm/figures/experiments_hidden_layers.png',
                                 f'ucca4bpm/figures/experiments_hidden_layers.pdf'
                             ]
                             )


plot_experiments_hidden_units()
plot_experiments_hidden_layers()
plot_experiments_lr()
plot_experiments_features()
