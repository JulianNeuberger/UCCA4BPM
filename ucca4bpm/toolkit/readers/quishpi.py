import argparse
import os

from ucca4bpm.toolkit.readers.base import BaseReader
from ucca4bpm.toolkit.readers.result import ReadResult, Document, ClassRange


class QuishpiReader(BaseReader):
    def __init__(self, remaining_arguments):
        super().__init__(remaining_arguments)

        parser = argparse.ArgumentParser(description='Subprogram for reading data files of Quishpi et al.')
        parser.add_argument(
            '--task',
            type=str,
            required=False,
            default='srl',
            help='Which task to read.',
            choices=['ssr', 'srl']
        )
        args, _ = parser.parse_known_args(remaining_arguments)

        self._mode = args.task

    def read(self, input_path: str) -> ReadResult:
        filenames = set([os.path.splitext(n)[0] for n in os.listdir(input_path) if not n.endswith('.py')])

        documents = []
        for n in filenames:
            with open(os.path.join(input_path, f'{n}.txt'), 'r') as text_file:
                document = Document(source=f'{n}.txt', text=text_file.read())
            with open(os.path.join(input_path, f'{n}.ann'), 'r') as annotation_file:
                ranges = {}
                for line in annotation_file:
                    annotation = line.split('\t')
                    # this is a semantic role, relations start with R, events with A
                    if annotation[0].startswith('T'):
                        clazz, start, stop = annotation[1].split(' ')
                        start, stop = int(start), int(stop)
                        assert document.text[start:stop] == annotation[2][:-1], f'{document.text[start:stop]} not matching {annotation[2][:-1]}'
                        r = ClassRange(start=start, stop=stop, clazz=clazz)
                        ranges[annotation[0]] = r
                    if annotation[0].startswith('A'):
                        new_clazz, other_id = annotation[1].split(' ')
                        ranges[other_id].clazz = new_clazz
                    # it's a sentence relation, starting with r
                    if annotation[0].startswith('R'):
                        clazz, arg1, arg2 = annotation[1].split(' ')
                        arg1 = arg1.split(':')[-1]
                        arg2 = arg2.split(':')[-1]
                        start = min(ranges[arg1].start, ranges[arg2].start)
                        stop = max(ranges[arg1].stop, ranges[arg2].stop)
                        ranges[annotation[0]] = ClassRange(clazz=clazz, start=start, stop=stop, is_global=True)
                if self._mode == 'ssr':
                    allowed_key_starts = ['R']
                else:
                    allowed_key_starts = ['A', 'T']
                document.ranges = [val for key, val in ranges.items() if key[0] in allowed_key_starts]
            documents.append(document)
        return ReadResult(documents=documents)

    @staticmethod
    def format_id() -> str:
        return 'quishpi'
