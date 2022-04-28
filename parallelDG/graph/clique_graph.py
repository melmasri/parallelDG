"""
Functions related to clique graph
"""

import networkx as nx


class CliqueGraph(nx.Graph):

    def __init__(self, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)
        self.num_graph_nodes = None

    def fresh_copy(self):
        """Return a fresh copy graph with the same data structure.

        A fresh copy has no nodes, edges or graph attributes. It is
        the same data structure as the current graph. This method is
        typically used to create an empty version of the graph.

        Notes
        -----
        If you subclass the base class you should overwrite this method
        to return your class of graph.
        """
        return CliqueGraph()

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        return super(CliqueGraph, self).add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        return super(CliqueGraph, self).add_edges_from(ebunch_to_add, **attr)

    def remove_edge(self, u, v):
        return super(CliqueGraph, self).remove_edge(u, v)

    def remove_edges_from(self, ebunch):
        self.log_nus = {}
        return super(CliqueGraph, self).remove_edges_from(ebunch)

    def add_node(self, n):
        return super(CliqueGraph, self).add_node(n)

    def add_nodes_from(self, nbunch_to_add):
        return super(CliqueGraph, self).add_nodes_from(nbunch_to_add)

    def add_separators_from(self, nbunch_to_add):
        return super(CliqueGraph, self).add_nodes_from(nbunch_to_add)
    
    def remove_node(self, n):
        return super(CliqueGraph, self).remove_node(n)

    def connected_component_vertices(self):
        return [list(c) for c in nx.connected_components(self)]

    def connected_components(self):
        return nx.connected_components(self)

    def to_graph(self):
        """ Returns the graph underlying the junction tree tree.

        Args:
            tree (NetworkX graph): A junction tree

        Returns:
            NetworkX graph
        """

        G = nx.Graph()
        for c in self.nodes():
            for n1 in set(c):
                if len(c) == 1:
                    G.add_node(n1)
                for n2 in set(c) - set([n1]):
                    G.add_edge(n1, n2)
        return G

    def tuple(self):
        return(frozenset(self.nodes()), frozenset([frozenset(e) for e in self.edges()]))

    def __hash__(self):
        return hash(self.tuple())


def subgraph_induced_by_subset(graph, s):
    """ Returns the subtree induced by the set s.

    Args:
       graph (NetworkX graph): A clique-graph
       s (set): Subset of the node in the underlying graph of T.
    """
    if len(s) == 0:
        return graph.copy()
    v_prime = {c for c in graph.nodes() if s <= c}
    return graph.subgraph(v_prime).copy()


def neighboring_cliques_node(graph, node):
    """ Return neighboring cliques for the node-induced clique-graph
    in  a dictionary. key:item pairs as (connector in node-induced):nei_clique
    Args:
      tree (NetwokrX) a clique graph 
      node (integer) a node
    """
    # TODO: add empty cliques
    if not type(node) is set and not type(node) is frozenset:
        node = frozenset([node])

    nei_cliques = dict()
    T = subgraph_induced_by_subset(graph, node)
    # nei_cliques (connect move)
    for subnode in T:           # subnode is not necessary a boundary clique
        for nei in graph.neighbors(subnode):
            if not node < nei:
                if subnode in nei_cliques.keys():
                    nei_cliques[subnode].append(nei)
                else:
                    nei_cliques[subnode] = [nei]
    return nei_cliques


def boundary_cliques_node(graph, node):
    """ Return boundary cliques for the node-induced clique-graph
    in  a dictionary. key:item pairs as (connector in node-induced):nei_clique
    Args:
      tree (NetwokrX) a clique graph 
      node (integer) a node
    """
    
    if not type(node) is set and not type(node) is frozenset:
        node = frozenset([node])

    bd_cliques = dict()
    T = subgraph_induced_by_subset(graph, node)
    # nei_cliques (connect move)
    for subnode in T:           # subnode is not necessary a boundary clique
        sep = separators(T, subnode)
        if len(sep) == 1:
            bd_cliques[subnode] = sep
    return bd_cliques


def separators(graph, clique):

    sep = set()
    for nei in graph.neighbors(clique):
        sep.add(nei & clique)

    return sep
