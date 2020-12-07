from typing import List

from tensorflow.keras.optimizers import Adam

from ucca4bpm.models.train import run
from ucca4bpm.util.history import Run
from ucca4bpm.util.visualization import plot_f1_metrics

num_folds = 5
span_matching_modes = {'fragment': 'F1 Fragment'}


def run_feature_experiment():
    runs: List[Run] = []
    for feature_name in [
        'featureless',
        'google300',
        'glove25',
        'postags_fine'
    ]:
        experiment_name = f'our_approach_our_data_features'
        runs.extend((
            run(run_name=f'our_approach_our_data_sc_features{feature_name}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_sc_{feature_name}.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=5,
                optimizer=Adam(lr=1e-5),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_ssr_features{feature_name}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_ssr_{feature_name}.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=10,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_srl_features{feature_name}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_srl_{feature_name}.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=15,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_quishpi_features{feature_name}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_quishpi_{feature_name}.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=15,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=0,
                num_folds=num_folds),
            ))
    plot_f1_metrics(runs, 'Importance of Feature Choice', {'f1_fragment': 'F1 (Fragment)'})


def run_hidden_len_experiment():
    runs: List[Run] = []
    for hidden_len in [
        16,
        32,
        64,
        128
    ]:
        experiment_name = f'our_approach_our_data_hidden_len'
        runs.extend((
            run(run_name=f'our_approach_our_data_sc_hidden_len_{hidden_len}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_sc_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=hidden_len,
                num_epochs=5,
                optimizer=Adam(lr=1e-5),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_ssr_hidden_len_{hidden_len}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_ssr_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=hidden_len,
                num_epochs=10,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_srl_hidden_len_{hidden_len}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_srl_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=hidden_len,
                num_epochs=15,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_quishpi_hidden_len_{hidden_len}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_quishpi_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=hidden_len,
                num_epochs=15,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=0,
                num_folds=num_folds),
        ))
    plot_f1_metrics(runs, 'Importance of Hidden Units', {'f1_fragment': 'F1 (Fragment)'})


def run_hidden_layers_experiment():
    runs: List[Run] = []
    for hidden_layers in [
        0,
        2,
        4,
        8
    ]:
        experiment_name = f'our_approach_our_data_hidden_layers'
        runs.extend((
            run(run_name=f'our_approach_our_data_sc_hidden_layers_{hidden_layers}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_sc_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=5,
                optimizer=Adam(lr=1e-5),
                early_stopping=False,
                num_gcn_hidden_layers=hidden_layers,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_ssr_hidden_layers_{hidden_layers}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_ssr_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=10,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=hidden_layers,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_srl_hidden_layers_{hidden_layers}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_srl_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=15,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=hidden_layers,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_quishpi_hidden_layers_{hidden_layers}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_quishpi_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=15,
                optimizer=Adam(lr=5e-5),
                early_stopping=False,
                num_gcn_hidden_layers=hidden_layers,
                none_class_id=0,
                num_folds=num_folds),
            ))
    plot_f1_metrics(runs, 'Importance of Hidden Layers', {'f1_fragment': 'F1 (Fragment)'})


def run_lr_experiment():
    runs: List[Run] = []
    for lr in [
        5e-3,
        5e-4,
        5e-5,
        5e-6
    ]:
        experiment_name = f'our_approach_our_data_lr'
        lr_formatted = f'{lr}'.replace('.', '-')
        runs.extend((
            run(run_name=f'our_approach_our_data_sc_lr_{lr_formatted}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_sc_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=5,
                optimizer=Adam(lr=lr),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_ssr_lr_{lr_formatted}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_ssr_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=10,
                optimizer=Adam(lr=lr),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_srl_lr_{lr_formatted}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_qian_srl_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=15,
                optimizer=Adam(lr=lr),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=None,
                num_folds=num_folds),
            run(run_name=f'our_approach_our_data_quishpi_lr_{lr_formatted}',
                experiment_name=experiment_name,
                data_set_path=f'ucca4bpm/data/transformed/ours_quishpi_postags_fine.pickle',
                span_matching_modes=list(span_matching_modes.keys()),
                gcn_hidden_len=64,
                num_epochs=15,
                optimizer=Adam(lr=lr),
                early_stopping=False,
                num_gcn_hidden_layers=2,
                none_class_id=0,
                num_folds=num_folds),
        ))
    plot_f1_metrics(runs, 'Importance of Learning Rate', {'f1_fragment': 'F1 (Fragment)'})


run_lr_experiment()
run_feature_experiment()
run_hidden_layers_experiment()
run_hidden_len_experiment()
