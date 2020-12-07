import os

from ucca4bpm.toolkit.readers.result import ReadResult


class BaseWriter:
    def __init__(self, remaining_arguments):
        pass

    def dump(self, read_result: ReadResult, out_file_path: str):
        out = self.dumps(read_result)
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        with open(out_file_path, 'w', encoding='utf8') as f:
            f.write(out)

    def dumps(self, read_result: ReadResult):
        raise NotImplemented()

    @staticmethod
    def format_id() -> str:
        raise NotImplementedError()
