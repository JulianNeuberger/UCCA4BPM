from typing import Iterable


def normalize_tokens(tokens: Iterable[str]):
    return [t for t in tokens if t != '']

