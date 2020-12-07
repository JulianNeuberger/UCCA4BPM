import argparse

from ucca4bpm.toolkit.readers.result import ReadResult
from ucca4bpm.toolkit.writers.base import BaseWriter


class QianWriter(BaseWriter):
    def __init__(self, remaining_arguments):
        super().__init__(remaining_arguments)
        parser = argparse.ArgumentParser(description='Subprogram for writing data files that can be read by '
                                                     'Qian et al.')
        parser.add_argument(
            '--mode',
            type=str,
            required=True,
            choices=['sc', 'ssr', 'srl'],
            help='Which mode to write.'
        )
        parser.add_argument(
            '--fragment-classes',
            type=str,
            nargs='+',
            required=False,
            help='Names of classes, that define fragments.'
        )
        args, _ = parser.parse_known_args(remaining_arguments)

        self._fragment_classes = args.fragment_classes
        self._mode = args.mode

    @staticmethod
    def format_id() -> str:
        return 'qian'

    @staticmethod
    def _get_result_set(read_result: ReadResult):
        if read_result.documents is not None:
            return read_result.documents
        elif read_result.sentences is not None:
            return read_result.sentences
        else:
            raise ValueError('Neither documents nor sentences is set in the given read result.')

    def _dump_as_sc(self, read_result):
        result_set = self._get_result_set(read_result)
        lines = []
        possible_classes = {
            'Statement': 'template',
            'Activity': 'activity'
        }
        for ele in result_set:
            for r in ele.ranges:
                sentence = ele.text[r.start: r.stop]
                sentence = sentence.replace(', ', '')
                sentence = sentence.strip()
                clazz = r.clazz
                lines.append(f'{sentence}, {possible_classes[clazz]}')
        return '\n'.join(lines)

    def _dump_as_srl(self, read_result: ReadResult):
        assert self._fragment_classes is not None
        result_set = self._get_result_set(read_result)
        sentences = []
        for ele in result_set:
            ranges = list(ele.ranges)
            for r in ranges:
                if r.clazz in self._fragment_classes:
                    ele.ranges.remove(r)
                    sentence = ele.text[r.start: r.stop]
                    tokenized_sentence = sentence.split(' ')
                    offset = 0
                    sentence = sentence.replace(', ', ' ')
                    sentence = sentence.strip()
                    token_id = 0
                    for token in tokenized_sentence:
                        token_len = len(token)
                        if token_len == 0:
                            continue
                        token_clazz = ele.get_class_for_range(r.start + offset, r.start + offset + token_len)
                        sentences.append(f'{sentence}, {token_id}, {token_clazz}')
                        offset += token_len + 1  # +1 for whitespace
                        assert token_len != 0, f'Zero token in sentence "{sentence}"'
                        token_id += 1
        return '\n'.join(sentences)

    def _dump_as_ssr(self, read_result: ReadResult):
        assert self._fragment_classes is not None
        result_set = self._get_result_set(read_result)
        sentences = []
        for ele in result_set:
            for r in ele.ranges:
                if r.clazz in self._fragment_classes:
                    sentence = ele.text[r.start: r.stop].strip()
                    if ', ' in sentence:
                        print('WARN: skipping sentence as it contains a ", "')
                        continue
                    sentences.extend([
                        f'{sentence}, {c}, {"y" if c == r.clazz else "n"}'
                        for c in self._fragment_classes
                    ])
        return '\n'.join(sentences)

    def dumps(self, read_result: ReadResult):
        if self._mode == 'sc':
            return self._dump_as_sc(read_result)
        elif self._mode == 'srl':
            return self._dump_as_srl(read_result)
        elif self._mode == 'ssr':
            return self._dump_as_ssr(read_result)
        raise ValueError('Unknown mode.')
