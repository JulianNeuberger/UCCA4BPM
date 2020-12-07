import json


with open('quishpi_org.mrp') as f:
    counter = 0
    for l in f:
        found_in_this = False
        graph = json.loads(l)
        for node in graph['nodes']:
            if 'anchors' in node:
                if len(node['anchors']) > 1:
                    if found_in_this:
                        counter += 1
                    found_in_this = True
    print(counter)
