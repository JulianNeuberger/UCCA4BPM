import json


with open('ours_qian_sc.json') as sc_file, open('ours_qian_srl.json') as srl_file:
    sc_as_json = json.load(sc_file)
    srl_as_json = json.load(srl_file)

    srl_tasks = {t['data']['source']: t for t in srl_as_json}

    for sc_task in sc_as_json:
        task_source = sc_task['data']['source']
        if task_source not in srl_tasks:
            continue
        sc_completions = sc_task['completions']
        if len(sc_completions) == 0:
            print(f'WARN: no completions for task with source {task_source}')
            continue
        if len(sc_completions) > 1:
            print(f'WARN: more than one completions for task with source {task_source}, taking first one.')
        sc_completions = sc_completions[0]
        assert sc_task['data']['text'] == srl_tasks[task_source]['data']['text'], f"\n'{sc_task['data']['text']}'\n'{srl_tasks[task_source]['data']['text']}"
        for srl_completion in srl_tasks[task_source]['completions']:
            srl_completion['result'].extend(sc_completions['result'])

with open('ours_qian_srl.json', 'w') as f:
    json.dump(list(srl_tasks.values()), f)
