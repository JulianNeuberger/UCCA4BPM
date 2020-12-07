import os
from typing import Optional

from ucca4bpm.toolkit.readers.base import BaseReader
from ucca4bpm.toolkit.readers.result import ReadResult, Sentence, Token


class QianReader(BaseReader):
    @staticmethod
    def format_id() -> str:
        return 'qian'

    def read(self, input_path: str) -> ReadResult:
        sentences = []
        sent_idx = 0
        with open(input_path, 'r', encoding='utf8') as f:
            file_name = os.path.basename(input_path)
            mode = None
            cur_sentence: Optional[Sentence] = None
            for line_number, line in enumerate(f):
                if line.strip() == '':
                    continue
                mode_for_line = self._guess_qian_task(line)
                if mode is None:
                    mode = mode_for_line
                if mode_for_line != mode:
                    raise ValueError('Mode changed during parsing the file.')

                sent_idx += 1

                if mode == 'SRL':
                    sentence_text, token_idx, token_class = line.split(', ')
                    if cur_sentence is None or sentence_text != cur_sentence.text:
                        cur_sentence = Sentence(
                            text=sentence_text,
                            source=(file_name, line_number),
                            sentence_id=sent_idx,
                            tokens=[]
                        )
                        sentences.append(cur_sentence)
                    cur_sentence.tokens.append(
                        Token(text=sentence_text.split(' ')[int(token_idx)], clazz=token_class.strip())
                    )
                elif mode == 'SC':
                    sentence_text, clazz = line.split(', ')
                    sentences.append(Sentence(
                        text=sentence_text,
                        source=(file_name, line_number),
                        sentence_id=sent_idx,
                        clazz=clazz.strip(),
                        tokens=[Token(text=t) for t in sentence_text.split(' ') if t != ''])
                    )
                elif mode == 'SSR':
                    sentence_text, clazz, relevancy_flag = line.split(', ')
                    if relevancy_flag.strip().upper() == 'Y':
                        sentences.append(Sentence(
                            text=sentence_text,
                            source=(file_name, line_number),
                            sentence_id=sent_idx,
                            clazz=clazz.strip(),
                            tokens=[Token(text=t) for t in sentence_text.split(' ') if t != ''])
                        )
                else:
                    raise ValueError(f'Unknown mode {mode}.')
        return ReadResult(sentences=sentences)

    @staticmethod
    def _guess_qian_task(line: str) -> str:
        line = line.split(', ')
        if len(line) == 2:
            return 'SC'
        elif len(line) == 3:
            try:
                # srl data has an integer (token index) at pos 2
                int(line[1])
                return 'SRL'
            except ValueError:
                # there was a non-number at pos 2, ssr data has a class (str) there
                return 'SSR'
        raise ValueError(f'Unknown format for line "{line}"')
