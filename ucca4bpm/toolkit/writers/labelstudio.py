import json

from ucca4bpm.toolkit.readers.result import ReadResult, Sentence, Document
from ucca4bpm.toolkit.writers.base import BaseWriter


class LabelStudioWriter(BaseWriter):
    def dumps(self, read_result: ReadResult):
        if read_result.documents is not None:
            tasks = [self._dumps_document(d) for d in read_result.documents]
        elif read_result.sentences is not None:
            tasks = [self._dumps_sentence(s) for s in read_result.sentences]
        else:
            raise ValueError('Unsupported read output without document or sentences.')
        return json.dumps(tasks)

    @staticmethod
    def _dumps_sentence(sentence: Sentence):
        return {
            'text': sentence.text,
            'source': sentence.source
        }

    @staticmethod
    def _dumps_document(document: Document):
        return {
            'text': document.text,
            'source': document.source
        }

    @staticmethod
    def format_id() -> str:
        return 'labelstudio'
