from typing import Dict, Any, Iterable, List, Tuple, Union, Optional

import networkx as nx

IRRELEVANT_CLASS = 'inner'
NONE_CLASS = 'none'


def build_graph(raw_graph: Dict[str, Any]) -> Tuple[int, Union[nx.Graph, nx.DiGraph], str]:
    def get_node_attributes(n):
        attrs = {}
        if 'anchors' in n:
            node_text = []
            for anchor in n['anchors']:
                node_text.append(input_text[anchor['from']: anchor['to']])
            attrs['text'] = ' '.join(node_text)
        if 'classes' in n:
            possible_classes = [c for c in n['classes'] if c != NONE_CLASS]
            if len(possible_classes) == 1:
                attrs['class'] = n['classes'][0]
            elif len(possible_classes) == 0:
                attrs['class'] = NONE_CLASS
            else:
                possible_classes = list(set(possible_classes))
                if len(possible_classes) != 1:
                    print(
                        f'WARN: Got more than one "relevant" class in node with text {attrs["text"]}: {possible_classes}, will choose the last one.')
                attrs['class'] = possible_classes[-1]
        for pos_tags in ['coarse_pos_tags', 'fine_pos_tags']:
            if pos_tags in n:
                attrs[pos_tags] = n[pos_tags]

        return attrs

    def get_edge_attributes(e):
        attrs = {}
        if 'label' in e:
            attrs['label'] = e['label']
        if 'attributes' in e and 'remote' in e['attributes']:
            attrs['remote'] = True
        else:
            attrs['remote'] = False
        return attrs

    def add_master_node(nodes, edges):
        """
        For graph level classification, we need a master node, see
        https://github.com/tkipf/gcn/issues/4
        :return:
        """
        master_node = (len(nodes), {'class': raw_graph['class']})
        nodes.append(master_node)
        edges += [
            (
                n[0],
                master_node[0],
                {'label': '_global', 'remote': False}
            )
            for
            n in graph_nodes
        ]
        return nodes, edges

    input_text = raw_graph['input']
    graph_id = int(raw_graph['id'])

    graph_nodes = {
        node['id']: get_node_attributes(node)
        for
        node in raw_graph['nodes']
    }
    graph_edges = [
        (
            edge['source'],
            edge['target'],
            get_edge_attributes(edge)
        )
        for
        edge in raw_graph['edges']
    ]
    order_edges = []
    for node_id, attrs in graph_nodes.items():
        if (node_id + 1) in graph_nodes:
            # node has successor node
            if 'text' in attrs:
                # node has text, i.e. is terminal node
                if 'text' in graph_nodes[node_id + 1]:
                    # successor node is terminal as well
                    # order_edges.append((node_id, node_id + 1, {'label': '_forward', 'remote': False}))
                    order_edges.append((node_id + 1, node_id, {'label': '_backward', 'remote': False}))

    reverse_edges = [
        (target, source, attrs)
        for
        source, target, attrs
        in
        graph_edges
    ]

    graph_nodes = [
        (node_id, node_attrs)
        for node_id, node_attrs
        in graph_nodes.items()
    ]

    if 'class' in raw_graph and raw_graph['class']:
        # add a global master node, that will hold the graph class after prediction
        graph_nodes, graph_edges = add_master_node(graph_nodes, graph_edges)

    graph = nx.DiGraph()
    graph.add_nodes_from(graph_nodes)
    graph.add_edges_from(graph_edges)
    graph.add_edges_from(reverse_edges)
    graph.add_edges_from(order_edges)

    return graph_id, graph, input_text


def build_graphs(raw_graphs: Iterable[Dict[str, Any]]) -> List[Tuple[int, nx.DiGraph, str]]:
    return [build_graph(g) for g in raw_graphs]


def get_label_for_edge(edge_attr):
    label = 'UNKNOWN'
    if 'label' in edge_attr:
        label = edge_attr['label']
    if 'remote' in edge_attr and edge_attr['remote']:
        label += '*'
    return label


def get_node_class_for_node(node_attr, class_mapping: Dict[str, Optional[str]]):
    if 'class' in node_attr:
        clazz = node_attr['class']
        if clazz in class_mapping:
            clazz = class_mapping[clazz]
            if clazz is None:
                clazz = NONE_CLASS
        return clazz
        # if clazz in ('Business Object', 'Role',):
        #     #clazz = 'Entity'
        #     return clazz
        # if clazz in ('Negation', 'Entity', 'Business Object', 'Role',):
        #     return NONE_CLASS
        # if clazz == 'Event':
        #     return 'Task'
    return IRRELEVANT_CLASS
