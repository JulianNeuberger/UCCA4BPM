import argparse
import json
from typing import Optional, List

from ucca4bpm.toolkit.readers.base import BaseReader
from ucca4bpm.toolkit.readers.result import ReadResult, Sentence, ClassRange, Document


class LabelStudioReader(BaseReader):
    def __init__(self, remaining_arguments):
        super().__init__(remaining_arguments)

        parser = argparse.ArgumentParser(description='Subprogram for reading data files that were labeled in labelstudio.')
        parser.add_argument(
            '--allowed-sources',
            type=str,
            nargs='+',
            required=False,
            help='Names of allowed sources in label studio tasks.'
        )
        args, _ = parser.parse_known_args(remaining_arguments)

        self._valid_sources = args.allowed_sources

    def read(self, input_path: str) -> ReadResult:
        with open(input_path, encoding='utf8') as f:
            tasks = json.load(f)

        documents = []

        for task in tasks:
            source = task['data']['source'] if 'data' in task and 'source' in task['data'] else None
            if self._valid_sources is not None:
                if source is None:
                    continue
                elif source not in self._valid_sources:
                    continue

            completions = [c for c in task['completions'] if 'skipped' not in c or not c['skipped']]
            if len(completions) == 0:
                print(f'Skipping task with id {task["id"]}, since it has no completions.')
                continue
            # always taking the first one...
            completion = completions[0]
            doc = Document(text=task['data']['text'],
                           source=task['data']['source'])
            class_ranges = []
            for label_result in completion['result']:
                for label_clazz in label_result['value']['labels']:
                    r = ClassRange(start=label_result['value']['start'],
                                   stop=label_result['value']['end'],
                                   clazz=label_clazz)
                    class_ranges.append(r)
            doc.ranges = class_ranges
            documents.append(doc)

        return ReadResult(documents=documents)

    @staticmethod
    def format_id() -> str:
        return 'labelstudio'
