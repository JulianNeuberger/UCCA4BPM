import os

from ucca4bpm.toolkit.readers.base import BaseReader
from ucca4bpm.toolkit.readers.result import ReadResult, Sentence


class HanReader(BaseReader):
    def read(self, input_path: str) -> ReadResult:
        sentences = []
        with open(input_path, 'r', encoding='windows-1252') as f:
            for sentence_id, line in enumerate(list(f)[1:]):
                sentence_text = line.split(';')[5]
                sentences.append(Sentence(
                    text=sentence_text,
                    source=(os.path.basename(input_path), sentence_id),
                    sentence_id=sentence_id
                ))
        return ReadResult(sentences=sentences)

    @staticmethod
    def format_id() -> str:
        return 'han'
