import json
import re
import os


set_names = set([os.path.splitext(n)[0] for n in os.listdir('theirs') if not n.endswith('.py')])
set_names = [(f'{n}.txt', f'{n}.ann') for n in set_names]

for text_file_name, annotation_file_name in set_names:
    with open(annotation_file_name, 'r') as annotation_file:
        annotations = []
        for l in annotation_file:
            l = l.strip()
            annotation = l.split('\t')
            annotation[1] = annotation[1].split(' ')
            annotations.append(annotation)

    with open(text_file_name, 'r') as text_file:
        text = text_file.read()

        match = re.search(r'\s\s', text)
        while match is not None:
            for annotation in annotations:
                if annotation[0].startswith('T'):
                    start = int(annotation[1][1])
                    end = int(annotation[1][2])
                    if match.start() < start:
                        annotation[1][1] = str(start - 1)
                        annotation[1][2] = str(end - 1)
            text = re.sub(r'\s\s', ' ', text, count=1)
            match = re.search(r'\s\s', text)

    with open(annotation_file_name, 'w', encoding='utf8') as f:
        for annotation in annotations:
            annotation[1] = ' '.join(annotation[1])
        annotations = ['\t'.join(a) for a in annotations]
        f.write('\n'.join(annotations))

    with open(text_file_name, 'w', encoding='utf8') as f:
        f.write(text)
