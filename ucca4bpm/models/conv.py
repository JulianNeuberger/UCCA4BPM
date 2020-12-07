from typing import Optional

from dgl.nn.tensorflow import RelGraphConv
from tensorflow.keras import layers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Dropout

from tensorflow.keras.backend import count_params

import numpy as np


class DGLModel(layers.Layer):
    def __init__(self, model_layers: list):
        super().__init__()
        self._layers = model_layers

    def call(self, g, h, r, norm=None, training=False):
        for name, layer in enumerate(self._layers):
            if type(layer) == RelGraphConv:
                h = layer(g, h, r, norm)
            else:
                h = layer(h, training=training)
        return h

    def get_num_trainable_params(self):
        num_params = 0
        layer: layers.Layer
        for layer in self._layers:
            param_counts = [count_params(n) for n in layer.trainable_variables]
            num_params += np.sum(param_counts)
        return num_params


def build_model(features_len, hidden_len, target_len, num_edge_types, num_bases: Optional[int] = None,
                num_hidden_layers: int = 1, dropout=.0) -> DGLModel:
    assert num_bases is None or num_bases > 1, 'num_bases must at least be 1 or None, if all edge types should be used.'

    _layers = [RelGraphConv(in_feat=features_len, out_feat=hidden_len,
                            num_rels=num_edge_types, num_bases=num_bases,
                            regularizer='basis',
                            activation=relu, self_loop=True, bias=True)]

    if dropout != 0.0:
        _layers.append(Dropout(rate=dropout))

    for _ in range(num_hidden_layers):
        _layers.append(RelGraphConv(in_feat=hidden_len, out_feat=hidden_len,
                                    num_rels=num_edge_types, num_bases=num_bases,
                                    regularizer='basis',
                                    activation=relu, self_loop=True, bias=True))
        if dropout != 0.0:
            _layers.append(Dropout(rate=dropout))

    _layers.append(RelGraphConv(in_feat=hidden_len, out_feat=target_len,
                                num_rels=num_edge_types, num_bases=num_bases,
                                regularizer='basis',
                                activation=softmax, self_loop=True, bias=True))

    return DGLModel(_layers)
