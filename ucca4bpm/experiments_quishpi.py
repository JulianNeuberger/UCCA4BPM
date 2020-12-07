import datetime
import itertools

from tensorflow.keras.optimizers import Adam

from ucca4bpm.models.train import run
from ucca4bpm.util.visualization import plot_f1_metrics

num_folds = 5

# OUR DATA

span_matching_modes = {'exact': 'F1 Exact', 'fragment': 'F1 Fragment', 'partial': 'F1 Partial'}
experiment_name = f'our_approach_quishpi_data'
plot_f1_metrics(
    [
        # run(run_name=f'quishpi_their_data',
        #     experiment_name=experiment_name,
        #     data_set_path='ucca4bpm/data/transformed/quishpi_postags_fine.pickle',
        #     span_matching_modes=list(span_matching_modes.keys()),
        #     gcn_hidden_len=64,
        #     num_epochs=15,
        #     optimizer=Adam(lr=5e-5),
        #     early_stopping=False,
        #     num_gcn_hidden_layers=2,
        #     none_class_id=None,
        #     num_folds=num_folds),

        run(run_name=f'quishpi_our_data',
            experiment_name=experiment_name,
            data_set_path='ucca4bpm/data/transformed/ours_quishpi_postags_fine.pickle',
            span_matching_modes=list(span_matching_modes.keys()),
            gcn_hidden_len=64,
            num_epochs=15,
            optimizer=Adam(lr=5e-5),
            early_stopping=False,
            num_gcn_hidden_layers=2,
            none_class_id=None,
            num_folds=num_folds),
    ],
    'Our Approach in Optimal Configuration on Data by Quishpi et al.',
    {f'f1_{mode}': label for mode, label in span_matching_modes.items()})
