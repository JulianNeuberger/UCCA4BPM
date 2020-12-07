import os

from ucca4bpm.toolkit.readers.base import BaseReader
from ucca4bpm.toolkit.readers.result import ReadResult, Sentence, Document


class DocumentsReader(BaseReader):
    def read(self, input_path: str) -> ReadResult:
        sentences = []
        documents = []
        sentence_id = 0
        for f_path in os.listdir(input_path):
            with open(os.path.join(input_path, f_path), 'r', encoding='utf8') as f:
                document = f.read()
                documents.append(Document(text=document, source=f_path))
                for line_number, sentence in enumerate(document.splitlines()):
                    sentences.append(Sentence(text=sentence, sentence_id=sentence_id, source=(f_path, line_number)))
                    sentence_id += 1

        return ReadResult(documents=documents, sentences=sentences)

    @staticmethod
    def format_id() -> str:
        return 'documents'
