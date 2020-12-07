import json

with open('quishpi_org.mrp') as f:
    for g in f:
        g = json.loads(g)
        text = g['input']
        formatted = []
        for node in g['nodes']:
            if 'anchors' in node:
                for anchor in node['anchors']:
                    formatted.append('|')
                    formatted.append(text[anchor['from']: anchor['to']])
                    formatted.append(f'[{node["class"]}]')
                    formatted.append('| ')
        print(f'{g["id"]}: {"".join(formatted)}')
