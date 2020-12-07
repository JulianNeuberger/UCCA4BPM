from typing import Union

import networkx as nx
import numpy as np
import torch
from gensim import downloader as api
from gensim.models import Word2Vec

from transformers import BertTokenizer, BertModel


class BaseNodeFeatureBuilder:
    def __call__(self, node_id: int, node_attrs: dict, graph: nx.DiGraph) -> Union[int, float, np.ndarray]:
        raise NotImplementedError()


class DebugFeatureBuilder(BaseNodeFeatureBuilder):
    def __init__(self):
        print('WARNING: You are using the debugging feature builder, '
              'which will result in the node target class being the node feature!!')

    def __call__(self, node_id: int, node_attrs: dict, graph: nx.DiGraph) -> Union[int, float, np.ndarray]:
        return node_attrs['class_one_hot']


class IdNodeFeatureBuilder(BaseNodeFeatureBuilder):
    def __call__(self, node_id: int, node_attrs: dict, graph: nx.DiGraph) -> Union[int, float, np.ndarray]:
        return node_id


class PosFeatureBuilder(BaseNodeFeatureBuilder):
    def __init__(self, mode):
        self._mode = mode

    def __call__(self, node_id: int, node_attrs: dict, graph: nx.DiGraph) -> Union[int, float, np.ndarray]:
        if self._mode == 'coarse-pos':
            return node_attrs['coarse_pos_tags_encoded']
        elif self._mode == 'fine-pos':
            return node_attrs['coarse_pos_tags_encoded']
        raise ValueError(f'Unknown mode: "{self._mode}"')


class BertFeatureBuilder(BaseNodeFeatureBuilder):
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self._model = BertModel.from_pretrained('bert-base-cased')

    def _get_unknown_vector(self):
        return np.zeros(768)

    def __call__(self, node_id: int, node_attrs: dict, graph: nx.DiGraph) -> Union[int, float, np.ndarray]:
        if 'text' not in node_attrs or node_attrs['text'] == '':
            return self._get_unknown_vector()
        text = node_attrs['text']
        tokens = self._tokenizer.tokenize(text, add_special_tokens=True)
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        print(f'{tokens} = {token_ids}')
        input_ids = torch.tensor(token_ids).unsqueeze(0)
        outputs = self._model(input_ids)
        # pooled output is a tensor of (1, embedding_size)
        pooled_output = 1
        embedding: torch.Tensor = outputs[pooled_output]
        return embedding.detach().numpy().squeeze()


class Word2VecFeatureBuilder(BaseNodeFeatureBuilder):
    SUPPORTED_MODELS = ['fasttext-wiki-news-subwords-300',
                        'conceptnet-numberbatch-17-06-300',
                        'word2vec-ruscorpora-300',
                        'word2vec-google-news-300',
                        'glove-wiki-gigaword-50',
                        'glove-wiki-gigaword-100',
                        'glove-wiki-gigaword-200',
                        'glove-wiki-gigaword-300',
                        'glove-twitter-25',
                        'glove-twitter-50',
                        'glove-twitter-100',
                        'glove-twitter-200']

    def __init__(self, model_name):
        self.model: Word2Vec = api.load(model_name)
        self.model.init_sims(replace=True)

    def __call__(self, node_id: int, node_attrs: dict, graph: nx.DiGraph) -> Union[int, float, np.ndarray]:
        if 'text' in node_attrs:
            text = node_attrs['text']
            if text is not '':
                return self._build_w2v_vector(text)
        return self._get_none_vector()

    @staticmethod
    def _post_process_vector(vec):
        return np.append(vec, 1.0)

    def _get_unknown_vector(self):
        return np.zeros(self.model.vector_size)

    def _get_none_vector(self):
        return np.ones(self.model.vector_size)

    @staticmethod
    def _normalize_vector(vec: np.ndarray):
        vec = vec / np.linalg.norm(vec)
        vec += 1.
        vec /= 2.
        return vec

    def _build_w2v_vector(self, text):
        tokens = text.split(' ')
        ret = []
        for token in tokens:
            if token in self.model:
                # vec = np.random.random(self.model.vector_size)
                vec = self.model.wv.word_vec(token, use_norm=True).copy()
                # vec = self._normalize_vector(vec)
                ret.append(vec)
            else:
                print()
                print(f'WARN: Word not in corpus: {token}')

        if len(ret) == 0:
            # every single token in this node is unknown, treat as if it had no text at all
            # TODO: experiment with a different "special" vector, so the model can distinguish between empty and unknown
            return self._get_unknown_vector()
        else:
            ret = np.array(ret)
            ret = np.mean(ret, axis=0)

            if np.isnan(ret).any():
                print(f'WARN: Found a nan value in embedding: {ret}')

            # if np.any(np.where(ret < 0)):
            #     print(f'WARN: Found negative value in embedding: {ret}')

            return ret


class ConcatFeatureBuilder(BaseNodeFeatureBuilder):
    def __init__(self):
        self._w2v_feature_builder = Word2VecFeatureBuilder('word2vec-google-news-300')
        self._id_feature_builder = IdNodeFeatureBuilder()

    def __call__(self, node_id: int, node_attrs: dict, graph: nx.DiGraph) -> Union[int, float, np.ndarray]:
        return
