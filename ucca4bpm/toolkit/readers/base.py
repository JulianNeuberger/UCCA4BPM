from ucca4bpm.toolkit.readers.result import ReadResult


class BaseReader:
    def __init__(self, remaining_arguments):
        pass

    def read(self, input_path: str) -> ReadResult:
        raise NotImplementedError()

    @staticmethod
    def format_id() -> str:
        raise NotImplementedError()
