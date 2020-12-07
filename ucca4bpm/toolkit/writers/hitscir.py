import argparse
import json
from collections import OrderedDict
from datetime import datetime
from typing import List, Optional, Callable

import spacy_udpipe
from spacy.tokens import Span, Token
from spacy_udpipe import UDPipeLanguage

from ucca4bpm.toolkit.readers.result import ReadResult
from ucca4bpm.toolkit.readers.result import Sentence, Document
from ucca4bpm.toolkit.writers.base import BaseWriter


class HitScirWriter(BaseWriter):
    def __init__(self, remaining_arguments):
        super().__init__(remaining_arguments)
        parser = argparse.ArgumentParser(description='Subprogram for writing data files that will be processed '
                                                     'by the HIT-SCIR UCCA parser.')
        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='Path to the UDPipe model to use for processing the given text.'
        )
        args, _ = parser.parse_known_args(remaining_arguments)

        self._ud_model: UDPipeLanguage = spacy_udpipe.load_from_path('en', args.model)

    @staticmethod
    def format_id() -> str:
        return 'hitscir'

    def _dump_result(self, result, get_id: Callable[[int], int], org_sent=None, org_doc=None):
        ret_mrp = []
        ret_conll = []

        sent: Span
        token: Token
        for sent_id, sent in enumerate(list(result.sents)):
            tokens = list(sent)
            sentence_id = get_id(sent_id)
            mrp = {
                'id': str(sentence_id),
                'flavor': 1,
                'version': 1.1,
                'framework': 'ucca',
                'time': datetime.now().strftime('%Y-%m-%d'),
                'tops': [len(tokens)],
                'input': sent.text
            }
            if org_sent is not None and org_sent.clazz is not None:
                mrp['class'] = org_sent.clazz
            ret_mrp.append(mrp)

            conll = [f'#{sentence_id}']
            for token_id, token in enumerate(tokens):
                token_id += 1
                if token.dep_.lower().strip() == "root":
                    head_idx = 0
                else:
                    head_idx = token.head.i + 1 - token.sent[0].i
                token_as_conll = (
                    token_id,
                    token.text,
                    token.lemma_,
                    token.pos_,
                    token.tag_,
                    '_',
                    head_idx,
                    token.dep_,
                    '_',
                    self._build_misc(token, sent, token_id - 1, result, org_sent=org_sent, org_doc=org_doc,
                                     sent_id=sentence_id)
                )
                conll.append('\t'.join(map(str, token_as_conll)))
            ret_conll.append('\n'.join(conll))
        return ret_mrp, ret_conll

    def dumps(self, read_result: ReadResult):
        mrps = []
        conlls = []
        if read_result.sentences is not None:
            for sentence in read_result.sentences:
                if sentence.is_tokenized():
                    result = self._ud_model([[t.text for t in sentence.tokens]])
                else:
                    result = self._ud_model([sentence.text])
                new_mrp, new_conll = self._dump_result(result, get_id=lambda _: sentence.sentence_id, org_sent=sentence)
                mrps.extend(new_mrp)
                conlls.extend(new_conll)
        else:
            offset = 0
            for document in read_result.documents:
                result = self._ud_model(document.text)
                new_mrp, new_conll = self._dump_result(result, get_id=lambda sent_idx: sent_idx + offset,
                                                       org_doc=document)
                mrps.extend(new_mrp)
                conlls.extend(new_conll)
                offset += len(list(result.sents))

        mrps_formatted: List[str] = list(map(json.dumps, mrps))

        return self._augment_data(conlls, mrps_formatted)

    @staticmethod
    def _build_misc(token: Token, sent: Span, token_index: int, result,
                    org_sent: Optional[Sentence] = None,
                    org_doc: Optional[Document] = None,
                    sent_id=None):
        token_start = token.idx
        token_end = token.idx + len(token)
        ret = f'TokenRange={token_start - sent.start_char}:{token_end - sent.start_char}'
        ret += f'|CoarsePosTag={token.pos_}'
        ret += f'|FinePosTag={token.tag_}'
        if org_sent is not None:
            if org_sent.tokens is not None:
                # pre tokenized data, e.g. qian et al.
                org_token = org_sent.tokens[token_index]
                try:
                    next_org_token = org_sent.tokens[token_index + 1]
                    is_terminal = next_org_token.clazz != org_token.clazz
                except IndexError:
                    is_terminal = True
                assert org_token.clazz is None or org_sent.clazz is None, \
                    'Do not support both sentence wide classes and token-wise ones at the same time.'
                if org_token.clazz is not None:
                    ret += f'|NodeClass={org_token.clazz}'
                # elif org_sent.clazz is not None:
                #     ret += f'|NodeClass={org_sent.clazz}'
            else:
                # our data
                token_class = org_sent.get_class_for_range(token_start, token_end)
                ret += f'|NodeClass={token_class}'
        else:
            assert org_doc is not None
            assert org_doc.text[token_start:token_end] == result.doc.text[token_start: token_end], \
                f'Expected to see token "{org_doc.text[token_start:token_end]}", ' \
                f'but actually saw "{result.doc.text[token_start: token_end]}"'
            token_class = org_doc.get_class_for_range(token_start, token_end)
            ret += f'|NodeClass={token_class}'
        return ret

    @staticmethod
    def _augment_data(conlls: List[str], mrps: List[str]):
        augs = {}
        for conll in conlls:
            sentence_id = conll.split('\n')[0][1:]
            augs[sentence_id] = [line.split('\t') for line in conll.strip().split('\n')[1:]]

        ret: List[str] = []
        for mrp in mrps:
            mrp = json.loads(mrp, object_pairs_hook=OrderedDict)
            sentence_id = mrp['id']
            if sentence_id not in augs:
                print("id:{} not in companion".format(sentence_id))
            else:
                mrp['companion'] = augs[sentence_id]
                ret.append(json.dumps(mrp))
        return '\n'.join(ret)
