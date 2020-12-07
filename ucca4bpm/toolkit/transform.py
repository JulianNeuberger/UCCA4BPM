import argparse
from typing import Type, Dict

from ucca4bpm.toolkit.readers.base import BaseReader
from ucca4bpm.toolkit.readers.documents import DocumentsReader
from ucca4bpm.toolkit.readers.han import HanReader
from ucca4bpm.toolkit.readers.labelstudio import LabelStudioReader
from ucca4bpm.toolkit.readers.qian import QianReader
from ucca4bpm.toolkit.readers.quishpi import QuishpiReader
from ucca4bpm.toolkit.readers.raw import RawReader
from ucca4bpm.toolkit.writers.base import BaseWriter
from ucca4bpm.toolkit.writers.hitscir import HitScirWriter
from ucca4bpm.toolkit.writers.labelstudio import LabelStudioWriter
from ucca4bpm.toolkit.writers.qian import QianWriter


def main():
    readers: Dict[str, Type[BaseReader]] = {r.format_id(): r for r in
                                            [QianReader, RawReader, HanReader, DocumentsReader, LabelStudioReader,
                                             QuishpiReader]}
    writers: Dict[str, Type[BaseWriter]] = {w.format_id(): w for w in [HitScirWriter, LabelStudioWriter, QianWriter]}

    parser = argparse.ArgumentParser(description='Transformer from raw text to HIT SCIR format.')
    parser.add_argument(
        '--inputs',
        required=True,
        type=str,
        help='Path to the input file or directory (depending on given input format).'
    )
    parser.add_argument(
        '--outputs',
        required=True,
        type=str,
        help='Path to the output file or directory (depends on given output format).'
    )
    parser.add_argument(
        '--in-format',
        type=str,
        required=True,
        help='Format of input.',
        choices=[r for r in readers.keys()]
    )
    parser.add_argument(
        '--out-format',
        type=str,
        required=True,
        help='Format of output.',
        choices=[w for w in writers.keys()]
    )
    args, remaining_arguments = parser.parse_known_args()

    in_format = args.in_format
    if in_format not in readers:
        raise ValueError(f'Unsupported input file format "{in_format}"')
    reader = readers[in_format](remaining_arguments)

    read_result = reader.read(args.inputs)

    out_format = args.out_format
    if out_format not in writers:
        raise ValueError(f'Unsupported output file format "{out_format}"')
    writer = writers[out_format](remaining_arguments)

    writer.dump(read_result, args.outputs)


if __name__ == '__main__':
    main()
