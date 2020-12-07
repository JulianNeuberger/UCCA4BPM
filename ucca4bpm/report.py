import numpy as np

from ucca4bpm.util.visualization import build_run_history_from_path

runs = {
    'Ours on Quishpi Data': 'ucca4bpm/runs/our_approach_quishpi_data/quishpi_their_data.json',
    'Ours on Our Data ala Quishpi': 'ucca4bpm/runs/our_approach_quishpi_data/quishpi_our_data.json',

    'Quishpi on Quishpi Data': 'ucca4bpm/runs/quishpi_approach_quishpi_data/quishpi_on_quishpi_data.json',
    'Quishpi on Our Data ala Quishpi': 'ucca4bpm/runs/quishpi_approach_quishpi_data/quishpi_on_our_data.json',

    'Qian SC on our data': 'ucca4bpm/runs/qian_approach_our_data/ours_qian_sc.json',
    'Qian SSR on our data': 'ucca4bpm/runs/qian_approach_our_data/ours_qian_ssr.json',
    'Qian SRL on our data': 'ucca4bpm/runs/qian_approach_our_data/ours_qian_srl.json',

    'Ours SC on our data': 'ucca4bpm/runs/our_approach_qian_data/ours_sc.json',
    'Ours SSR on our data': 'ucca4bpm/runs/our_approach_qian_data/ours_ssr.json',
    'Ours SRL on our data': 'ucca4bpm/runs/our_approach_qian_data/ours_srl.json',

    'Qian on COR (CC)': 'ucca4bpm/runs/qian_approach_qian_data/COR_sc.json',
    'Qian on COR (CSR)': 'ucca4bpm/runs/qian_approach_qian_data/COR_ssr.json',
    'Qian on COR (SRL)': 'ucca4bpm/runs/qian_approach_qian_data/COR_srl.json',

    'Qian on MAM (CC)': 'ucca4bpm/runs/qian_approach_qian_data/MAM_sc.json',
    'Qian on MAM (CSR)': 'ucca4bpm/runs/qian_approach_qian_data/MAM_ssr.json',
    'Qian on MAM (SRL)': 'ucca4bpm/runs/qian_approach_qian_data/MAM_srl.json',

    'Ours on COR (CC)': 'ucca4bpm/runs/our_approach_qian_data/SC_COR.json',
    'Ours on COR (CSR)': 'ucca4bpm/runs/our_approach_qian_data/SSR_COR.json',
    'Ours on COR (SRL)': 'ucca4bpm/runs/our_approach_qian_data/SRL_COR.json',

    'Ours on MAM (CC)': 'ucca4bpm/runs/our_approach_qian_data/SC_MAM.json',
    'Ours on MAM (CSR)': 'ucca4bpm/runs/our_approach_qian_data/SSR_MAM.json',
    'Ours on MAM (SRL)': 'ucca4bpm/runs/our_approach_qian_data/SRL_MAM.json',
}

matching_modes = ['exact', 'partial', 'fragment']

for run_name, run_path in runs.items():
    run_history = build_run_history_from_path(run_path, run_name, matching_modes, None)
    metrics = {
        f'f1_{mode}': []
        for mode in matching_modes
    }
    for fold_history in run_history.fold_histories:
        epoch = fold_history.epochs[-1]
        for mode in matching_modes:
            metric_name = f'f1_{mode}'
            metrics[metric_name].append(epoch.metrics[metric_name])

    for key in metrics.keys():
        metrics[key] = np.array(metrics[key])

    formatted_metrics = [f'\n\t{metric_name} || Mean: {np.mean(metric_values):.4f} | Median: {np.median(metric_values):.4f}'
                         for metric_name, metric_values in metrics.items()]

    print(f'{run_name}: {",".join(formatted_metrics)}')
