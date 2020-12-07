import json


with open('ours_qian_srl.json') as f:
    for t in json.load(f):
        for c in t['completions']:
            for r in c['result']:
                print(t['data']['text'][r['value']['start']: r['value']['end']])
