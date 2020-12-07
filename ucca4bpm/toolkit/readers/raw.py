import os

from ucca4bpm.toolkit.readers.base import BaseReader
from ucca4bpm.toolkit.readers.result import ReadResult, Sentence, Document


class RawReader(BaseReader):
    @staticmethod
    def format_id() -> str:
        return 'raw'

    def read(self, input_path: str) -> ReadResult:
        with open(input_path, 'r', encoding='utf8') as f:
            content = f.read()

        doc_source = os.path.basename(input_path)
        documents = [Document(text=content, source=doc_source)]
        sentences = [
            Sentence(text=sentence, source=(doc_source, line_number), sentence_id=line_number)
            for line_number, sentence
            in enumerate(content.split('\n'))
        ]
        return ReadResult(documents=documents, sentences=sentences)
