import argparse
import json
import os
import pickle
from typing import List, Optional, Tuple, Any, Dict, Union

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
# import torch
from sklearn.utils import compute_class_weight
from tensorflow.python.framework.ops import EagerTensor

from ucca4bpm.data import transform
from ucca4bpm.data.feature_builder import BaseNodeFeatureBuilder, IdNodeFeatureBuilder, Word2VecFeatureBuilder, \
    DebugFeatureBuilder, BertFeatureBuilder, PosFeatureBuilder, ConcatFeatureBuilder


def process_mrp_to_networkx(data_set_path):
    with open(data_set_path, 'r', encoding='utf8') as f:
        raw_graphs = [json.loads(graph) for graph in f]
    nx_graphs = transform.build_graphs(raw_graphs)
    return nx_graphs


def normalize_adjacency(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten())
        a_norm = d.dot(adj).tocsr()
    return a_norm


def process_adjacency(a):
    d = np.array(a.sum(1)).flatten()
    d_inv = np.divide(1., d, out=np.zeros_like(d), where=d != 0)
    big_d_inv = sp.diags(d_inv)
    return sp.lil_matrix(big_d_inv.dot(a))


def get_common_info(graphs: List[Tuple[int, Union[nx.DiGraph, nx.Graph], str]],
                    class_mapping: Dict[str, str]):
    # find the maximum number of nodes for all graphs in our data set
    max_num_nodes = max(g[1].number_of_nodes() for g in graphs)

    # analyze the given graphs
    node_classes = list()  # dont use a set, so we can still use .index during the first iteration
    coarse_pos_tags = set()
    fine_pos_tags = set()
    edge_labels = list()
    ys = []
    node_texts = {}

    for idx, (graph_id, graph, _) in enumerate(graphs):
        attr: Dict[str, Any]
        node_texts[idx] = {}
        for node_id, attr in graph.nodes(data=True):
            node_class = transform.get_node_class_for_node(attr, class_mapping)
            if node_class != transform.IRRELEVANT_CLASS:
                if node_class not in node_classes:
                    node_classes.append(node_class)
                ys.append(node_classes.index(node_class))
            coarse_pos_tags.update(attr.get('coarse_pos_tags', [transform.IRRELEVANT_CLASS]))
            fine_pos_tags.update(attr.get('fine_pos_tags', [transform.IRRELEVANT_CLASS]))
            node_texts[idx][node_id] = attr.get('text', '')
        for _, _, attr in graph.edges(data=True):
            edge_label = transform.get_label_for_edge(attr)
            if edge_label not in edge_labels:
                edge_labels.append(edge_label)

    # get the class weights via sklearns heuristic
    classes = np.arange(len(node_classes))
    ys = np.asarray(ys)
    class_weights = compute_class_weight('balanced', classes=classes, y=ys)
    class_weights = {i: class_weights[i] for i in classes}

    # get auxiliary objects
    graph_ids = []
    graph_texts = []

    for i, (graph_id, graph, graph_text) in enumerate(graphs):
        graph_ids.append(graph_id)
        graph_texts.append(graph_text)

    return graph_ids, graph_texts, class_weights, node_classes, edge_labels, max_num_nodes, \
           list(coarse_pos_tags), list(fine_pos_tags), node_texts


def process_networkx_to_dgl(graphs: List[Tuple[int, Union[nx.DiGraph, nx.Graph], str]],
                            node_feature_builder: Optional[BaseNodeFeatureBuilder] = None,
                            class_mapping=None):
    if class_mapping is None:
        class_mapping = {}
    node_feature_len = None
    if node_feature_builder is None:
        node_feature_builder = IdNodeFeatureBuilder()

    # dgl doesn't properly work for one edge graphs, filter them out
    graphs = [(gid, g, gtext) for gid, g, gtext in graphs if g.number_of_edges() > 1]

    print('Building common features...')
    graph_ids, graph_texts, class_weights, node_classes, edge_classes, max_num_nodes, coarse_pos_tags, fine_pos_tags, \
        node_texts = get_common_info(graphs, class_mapping)

    print('Calculating graph features...')
    # convert edge labels to ids
    for i, (_, g, _) in enumerate(graphs):
        for u, v, old_attrs in g.edges(data=True):
            edge_class_id = edge_classes.index(transform.get_label_for_edge(old_attrs))
            g.edges[u, v].update({
                'class_one_hot': tf.one_hot(edge_class_id, depth=len(edge_classes), dtype=tf.float32),
                'class_ordinal': edge_class_id
            })
        for n_id, old_attrs in g.nodes(data=True):
            node_class = transform.get_node_class_for_node(old_attrs, class_mapping)
            if node_class != transform.IRRELEVANT_CLASS:
                node_class_id = node_classes.index(node_class)
                node_class_one_hot = tf.one_hot(node_class_id, depth=len(node_classes), dtype=tf.float32)
            else:
                node_class_id = -1
                node_class_one_hot = tf.zeros((len(node_classes),))
            pos_tag_attrs = {}
            for pos_tags, pos_tag_names in [('coarse_pos_tags', coarse_pos_tags), ('fine_pos_tags', fine_pos_tags)]:
                pos_tag_ids = [pos_tag_names.index(p) for p in old_attrs.get(pos_tags, [transform.IRRELEVANT_CLASS])]
                pos_tag_attrs[f'{pos_tags}_ordinal'] = pos_tag_ids
                pos_tag_attrs[f'{pos_tags}_encoded'] = tf.math.add_n([
                    tf.one_hot(i, depth=len(pos_tag_names))
                    for i in pos_tag_ids
                ])
            g.nodes[n_id].update({
                'class_one_hot': node_class_one_hot,
                'class_ordinal': node_class_id,
                'is_target': node_class not in [transform.IRRELEVANT_CLASS]
            })
            g.nodes[n_id].update(pos_tag_attrs)

            node_feature = node_feature_builder(n_id, old_attrs, g)
            if type(node_feature) in [int, float]:
                new_node_feature_len = 1
            elif type(node_feature) in [np.ndarray, tf.Tensor, EagerTensor]:
                new_node_feature_len = node_feature.shape[0]
            elif type(node_feature) in [list, set]:
                new_node_feature_len = len(node_feature)
            else:
                raise ValueError(f'Unsupported feature type {type(node_feature)}.')
            if node_feature_len is None:
                node_feature_len = new_node_feature_len
            else:
                assert node_feature_len == new_node_feature_len, 'Inconsistent node feature lengths. Make sure the ' \
                                                                 'FeatureBuilder always returns the same size features.' \
                                                                 f'new: {new_node_feature_len} vs old: {node_feature_len}'
            g.nodes[n_id].update({
                'feature': node_feature,
            })
        print(f'\rDone with graph {i + 1}/{len(graphs)}', end='')
    print()

    print('Converting NetworkX graphs to DGL graphs...')
    dgl_graphs: Dict[int, dgl.DGLHeteroGraph] = {}
    for i, (g_id, g, _) in enumerate(graphs):
        dgl_graph: dgl.DGLHeteroGraph = dgl.from_networkx(
            g,
            edge_attrs=['class_one_hot', 'class_ordinal'],
            node_attrs=['class_one_hot', 'class_ordinal', 'is_target', 'feature']
        )
        dgl_graphs[i] = dgl_graph
        print(f'\rDone with graph {i + 1}/{len(graphs)}', end='')
    print()

    return {
        'ids': np.array(graph_ids),
        'texts': graph_texts,
        'class_weights': class_weights,
        'node_classes': node_classes,
        'edge_classes': edge_classes,
        'max_num_nodes': max_num_nodes,
        'node_feature_len': node_feature_len,

        'dgl_graphs': dgl_graphs,

        'node_texts': node_texts
    }


def main():
    parser = argparse.ArgumentParser(description='Processing utility for our datasets.')
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help='Path to the dataset to process.'
    )
    parser.add_argument(
        '--target',
        required=True,
        type=str,
        help='Path to the target file.'
    )
    parser.add_argument(
        '--features',
        type=str,
        help='Node features to build.',
        default='none',
        choices=['none', *Word2VecFeatureBuilder.SUPPORTED_MODELS, 'debug', 'bert', 'fine-pos', 'coarse-pos', 'concat']
    )
    parser.add_argument(
        '--mappings',
        nargs='*',
        required=False,
        help='One or many class mappings in the form <original>:<target>, '
             'e.g. to change all Events to Tasks use Event:Task'
    )
    args = parser.parse_args()

    feature_builder: BaseNodeFeatureBuilder
    if args.features == 'none' or args.features == 'None':
        feature_builder = IdNodeFeatureBuilder()
    elif args.features == 'debug':
        feature_builder = DebugFeatureBuilder()
    elif args.features == 'bert':
        feature_builder = BertFeatureBuilder()
    elif args.features in ['fine-pos', 'coarse-pos']:
        feature_builder = PosFeatureBuilder(args.features)
    elif args.features in Word2VecFeatureBuilder.SUPPORTED_MODELS:
        feature_builder = Word2VecFeatureBuilder(args.features)
    elif args.features == 'concat':
        feature_builder = ConcatFeatureBuilder()
    else:
        raise ValueError(f'Unknown feature builder "{args.features}"')

    class_mapping = {}
    if args.mappings:
        for mapping in args.mappings:
            source, target = mapping.split(':')
            if target == '':
                target = None
            class_mapping[source] = target
        print(f'Using class mapping {class_mapping}')

    print(f'Converting {args.dataset} to networkx graphs...')
    transformed_graphs = process_mrp_to_networkx(args.dataset)
    print('Done!')
    data = process_networkx_to_dgl(transformed_graphs, node_feature_builder=feature_builder, class_mapping=class_mapping)

    pickled = pickle.dumps(data)
    print(f'Writing approximately {len(pickled) / 1e6:.1f}MB of processed data to disk...')
    os.makedirs(os.path.dirname(args.target), exist_ok=True)
    with open(args.target, 'wb') as out_file:
        out_file.write(pickled)

    print('Done!')


if __name__ == '__main__':
    main()
