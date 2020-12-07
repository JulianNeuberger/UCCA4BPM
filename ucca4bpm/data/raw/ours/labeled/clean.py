import json
import re
import copy


file_path = 'ours_qian_srl.json'
with open(file_path, 'r', encoding='utf8') as f:
    docs = json.load(f)
    updated_docs = []
    tokens_before = []
    for doc in docs:
        for completion in doc['completions']:
            for result in completion['result']:
                tokens_before.append(doc['data']['text'][result['value']['start']: result['value']['end']])
    print(f'Before: {tokens_before}')
    for doc in docs:
        doc['data']['text'] = doc['data']['text'].replace('\n', ' ')
        before_len = len(doc['data']['text'])
        doc['data']['text'] = re.sub(r'[(),"\'/\\_\-]', ' ', doc['data']['text'])
        assert len(doc['data']['text']) == before_len
        completions = copy.deepcopy(doc['completions'])
        for m in re.finditer(r'[ ]{2,}', doc['data']['text']):
            for c_idx, completion in enumerate(completions):
                for r_idx, result in enumerate(completion['result']):
                    match_len = m.end() - m.start()
                    if result['value']['start'] >= m.start():
                        doc['completions'][c_idx]['result'][r_idx]['value']['start'] -= match_len - 1
                    if result['value']['end'] >= m.end():
                        doc['completions'][c_idx]['result'][r_idx]['value']['end'] -= match_len - 1

        doc['data']['text'] = re.sub(r'[ ]{2,}', ' ', doc['data']['text'])

        updated_docs.append(doc)
    tokens_after = []
    for doc in docs:
        for completion in doc['completions']:
            for result in completion['result']:
                tokens_after.append(doc['data']['text'][result['value']['start']: result['value']['end']])
    print(f'After:  {tokens_after}')

    print(f'Diff:   {[t for t in tokens_after if t not in tokens_before]}')

with open(file_path, 'w', encoding='utf8') as f:
    json.dump(updated_docs, f)
