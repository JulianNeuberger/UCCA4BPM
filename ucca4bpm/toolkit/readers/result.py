from typing import List, Optional, Tuple


class ReadResult:
    def __init__(self,
                 documents: Optional[List['Document']] = None,
                 sentences: Optional[List['Sentence']] = None):
        self.documents = documents
        self.sentences = sentences


class Document:
    def __init__(self, text, source: str, ranges: Optional[List['ClassRange']] = None):
        self.text = text
        self.source = source
        self.ranges = ranges

    def get_class_for_range(self, start: int, stop: int, default_class='none'):
        assert self.ranges is not None
        r: ClassRange
        for r in self.ranges:
            if r.is_in_range(start, stop):
                return r.clazz
        return default_class


class Sentence:
    def __init__(self, text, source: Tuple[str, int], sentence_id: int, clazz: Optional[str] = None,
                 tokens: Optional[List['Token']] = None, ranges: Optional[List['ClassRange']] = None):
        self.text = text
        self.sentence_id = sentence_id
        self.source = source
        self.clazz = clazz
        self.tokens = tokens
        self.ranges = ranges

    def get_class_for_range(self, start: int, stop: int, default_class='none'):
        assert self.ranges is not None
        r: ClassRange
        for r in self.ranges:
            if r.is_in_range(start, stop):
                return r.clazz
        return default_class

    def is_tokenized(self):
        return self.tokens is not None


class Token:
    def __init__(self, text, clazz: Optional[str] = None):
        self.text = text
        self.clazz = clazz


class ClassRange:
    def __init__(self, clazz: str, start: int, stop: int, is_global=False):
        self.start = start
        self.stop = stop
        self.clazz = clazz
        self.is_global = is_global

    def is_in_range(self, start: int, stop: int):
        return start >= self.start and stop <= self.stop
