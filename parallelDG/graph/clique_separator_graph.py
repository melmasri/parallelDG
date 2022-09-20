import networkx as nx
from parallelDG.graph import junction_tree as jtlib
from networkx.drawing.nx_pydot import graphviz_layout
import collections
import numpy as np
 
class CliqeSeparator(nx.DiGraph):

    def __init__(self, data=None, **attr):
        nx.DiGraph.__init__(self, data, **attr)
        self.separators = None
        self.num_graph_nodes = None
        self.num_components = None
        self.num_blocks = None

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
        return CliqeSeparator()

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self.separators = None
        return super(CliqeSeparator, self).add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        self.separators = None
        return super(CliqeSeparator, self).add_edges_from(ebunch_to_add, **attr)

    def remove_edge(self, u, v):
        self.separators = None
        return super(CliqeSeparator, self).remove_edge(u, v)

    def remove_node(self, n):
        self.separators = None
        return super(CliqeSeparator, self).remove_node(n)

    def remove_edges_from(self, ebunch):
        self.separators = None
        return super(CliqeSeparator, self).remove_edges_from(ebunch)

    def get_separators(self):
        if self.separators is None:
            self.separators = [node for node in self.nodes() if self.out_degree(node)]
        return self.separators

    def get_cliques(self):
        cliques = [node for node in self.nodes() if not self.out_degree(node)]
        return cliques

    def to_graph(self):
        """ Returns the graph underlying the clique-separator graph.

        Returns:
            NetworkX graph
        """

        G = nx.Graph()
        cliques = self.get_cliques()
        for c in cliques:
            for n1 in set(c):
                if len(c) == 1:
                    G.add_node(n1)
                for n2 in set(c) - set([n1]):
                    G.add_edge(n1, n2)
        return G
    
    def plot(self, **args):
        #pos = nx.planar_layout(cs)
        #pos = nx.spring_layout(self)
        pos = graphviz_layout(self, prog="dot")
        no_frozen_labels = {node:','.join(map(str, sorted(tuple(node)))) for node in self.nodes()}
        cliques = {clique:"tab:blue" for clique in self.get_cliques()}
        separators = {sep:"tab:red" for sep in self.get_separators()}
        cliques.update(separators)
        color_map = [cliques.get(n) for n in self.nodes()]
        
        options = {
            "font_size": 12,
            "node_color": color_map,
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1,
            'node_size': [ (len(node) + 1) * 600 for node in self.nodes()],
            'alpha' : 0.9,
            "edgecolors": "tab:gray"    
        }
        nx.draw(self,
                pos=pos,
                labels = no_frozen_labels,
                with_labels=True, **options)

def clique_separator_graph(jt):
    """ Returns the clique-separator graph as in Ibarra (2009)

    Args:
        jt (NetworkX graph): A junction tree

    Returns:
        DiGrpah (NetworkX graph): the clique-separator graph
    """
    cs  = nx.DiGraph()
    cs.add_nodes_from(list(jt.nodes()), clique = True)
    cs.add_node(frozenset([]), clique = False)
    sep_list = []
    for s in jt.edges():
       sep = s[0] & s[1]
       sep_list.append(sep)
    cs.add_nodes_from(sep_list, clique=False)
    cs.add_edges_from(sep2clq_edges(jt))
    ## adding blocks
    k = 0
    attr_dict = {}
    for comp in nx.weakly_connected_components(cs): 
        if frozenset([]) not in comp: 
            for node in comp: 
                attr_dict[node] = {"block" : k}
            k += 1
    for node in nx.neighbors(cs, frozenset([])):
        attr_dict[node] = {"block" : k}
        k += 1
    nx.set_node_attributes(cs, attr_dict)
    cs.add_edges_from(sep2sep_edges(jt))

    connected_components = jtlib.forest_induced_by_sep(jt, frozenset([])).connected_component_vertices()

    # adding cliques
    attr_dict2 = {}
    c = 0
    for comp in connected_components:
        for node in comp:
            attr_dict2[node] = {'component' : c}
        c +=1
    nx.set_node_attributes(cs, attr_dict2)

    CS = CliqeSeparator()
    CS.add_nodes_from(cs.nodes().data())
    CS.add_edges_from(cs.edges())
    CS.num_components = len(connected_components)
    CS.num_graph_nodes = jt.num_graph_nodes
    CS.num_blocks = k
    return CS


def unique_link(node1, node2, tree):
    """ Returns True if node1 < node2 and no other node n st. node1 < n < node2

    Args:
        node1: first node of tree
        node2: second node of tree
        tree (NetworkX graph): A junction tree

    Returns:
        logical: True, False
    """
    unique = True
    for d in tree.edges():
        sep = d[0] & d[1]
        if node1 < sep and sep < node2: 
            unique = False
            break
    return unique

def sep2clique_edge(sep, tree):
    """ Returns the edge to add from separator to clique

    Args:
        sep: a separator 
        tree (NetworkX graph): A junction tree

    Returns:
        list:  [(from, to), (from, to)]
    """
    edges_to_add = []
    for clq in tree.nodes():
        if sep < clq: 
            unique = unique_link(sep, clq, tree)
            if unique: 
                edges_to_add.append((sep, clq))
    return edges_to_add


def sep2sep_edge(sep, tree):
    """ Returns the edge to add from separator to separator

    Args:
        sep: a separator 
        tree (NetworkX graph): A junction tree

    Returns:
        list:  [(from, to), (from, to)]
    """
    edges_to_add = []
    for e in tree.edges():
        sep1 = e[0] & e[1]
        if sep < sep1: 
            unique = unique_link(sep, sep1, tree)
            if unique: 
                edges_to_add.append((sep, sep1))
    return edges_to_add


def sep2clq_edges(tree):
    """ Returns the separator to clique edges

    Args:
        tree (NetworkX graph): A junction tree

    Returns:
        list:  [(from, to), (from, to)]
    """
    edges_to_add = []
    for s in tree.edges():
        sep = s[0] & s[1]
        edges_to_add += sep2clique_edge(sep, tree)
    return edges_to_add


def sep2sep_edges(tree):
    """ Returns the separator to separator edges

    Args:
        tree (NetworkX graph): A junction tree

    Returns:
        list:  [(from, to), (from, to)]
    """
    edges_to_add = []
    for s in tree.edges():
        sep = s[0] & s[1]
        edges_to_add += sep2sep_edge(sep, tree)
    edges_to_add += sep2sep_edge(frozenset([]), tree)
    return edges_to_add

    

def subgraph_induced_by_subset(graph, s):
    """ Returns the subtree induced by the set s.

    Args:
       graph (NetworkX graph): A clique-separator graph
       s (set): Subset of the node in the underlying graph
    """
    if len(s) == 0:
        return graph.copy()
    v_prime = {c for c in graph.nodes() if s <= c}
    return graph.subgraph(v_prime)


def leaf_nodes(graph):
    """ Returns a set of the leaf nodes of graph
    Args:
      graph (NetworkX graph): a clique-separator graph
    """
    clq = graph.get_cliques()
    return [x for x in clq if graph.in_degree(x) == 1]


def descendants_in_blocks(graph, descendants):
    blocks = {b: [] for b in range(graph.num_blocks)}
    for sep, pathes in descendants.items(): 
        for path in pathes:
            end_node = path[-1]
            block = graph.node[end_node]['block']
            blocks[block].append(path)
    return blocks


def neighboring_nodes(graph, subset):
    """ Return neighboring nodes, to the induced graph

    Args:
      graph (NetwokrX) a clique-separator graph
      subset (frozenset) indeuction subset
      empty_node (bool) if empty cliques should be included
    """
    ancestor_descendant_pair = dict()      # format {clique: [path of separators], clique: [path of separators], ...}
    induced_graph = subgraph_induced_by_subset(graph, subset)
    cliques = induced_graph.get_cliques()
    ancestors_to_induced_graph = set()
    for n in induced_graph.nodes(): 
        for p in graph.predecessors(n):
            if not p & subset: 
                ancestors_to_induced_graph.add(p)
    direct_clqs = clique_successors(graph, ancestors_to_induced_graph, subset)
    direct_sep = separator_predecessors(graph,direct_clqs, subset)
    skip_nodes = direct_sep - ancestors_to_induced_graph
    component = induced_graph.nodes[cliques[0]]['component']
    for ans in list(ancestors_to_induced_graph):
        if ans: 
            ancestor_descendant_pair[ans] =  dfs_path2cliques_subgraph(graph,
                                                                       ans,
                                                                       subset,
                                                                       skip_nodes)
        else:
            pathes =  dfs_path2cliques_component(graph, ans, component)
            components = organize_into_components(graph, pathes)
            ancestor_descendant_pair[ans] =  sample_one_path(components)
    ## removing paths to cliques withing subset s
    #return ancestor_descendant_pair
    to_blocks = descendants_in_blocks(graph, ancestor_descendant_pair)
    return to_blocks



def dfs_path2cliques(graph, source):
    """ Return the DFS paths to cliques from the source.

    Args:
      graph (NetwokrX) a clique-separator graph
      source (node) soruce node

    Return: 
        [(soruce, to 1, ..., to final), (soruce, to 1, ..., to final)]
    """
    pathes = []
    path = [source]
    for edge in nx.dfs_edges(graph, source):
        if edge[0] == path[-1]:
            # node of path
            path.append(edge[1])
        else:
            # new path
            pathes.append(path)
            search_index = 2
            while search_index <= len(path):
                if edge[0] == path[-search_index]:
                    path = path[:-search_index + 1] + [edge[1]]
                    break
                search_index += 1
            else:
                raise Exception("Wrong path structure?", path, edge)
    pathes.append(path)
    return pathes



def dfs_path2cliques_subgraph(graph, source, subset):
    """ Return the DFS paths to cliques from the source.

    Args:
      graph (NetwokrX) a clique-separator graph
      source (node) soruce node
      subset (node) branch to exclude

    Return: 
        [(soruce, to 1, ..., to final), (soruce, to 1, ..., to final)]
    """
    pathes = []
    path = [source]
    for edge in nx.dfs_edges(graph, source, depth_limit=1):
        if edge[1] & subset:
            continue
        if edge[0] == path[-1]:
            # node of path
            path.append(edge[1])
        else:
            # new path
            if graph.node[path[-1]]['clique']:
                pathes.append(path)
            search_index = 2
            while search_index <= len(path):
                if edge[0] == path[-search_index]:
                    path = path[:-search_index + 1] + [edge[1]]
                    break
                search_index += 1
            else:
                raise Exception("Wrong path structure?", path, edge)
    if graph.node[path[-1]]['clique']:
        pathes.append(path)
    return pathes


def dfs_path2cliques_subgraph(graph, source, subset, skip_nodes):
    """ Return the DFS paths to cliques from the source.

    Args:
      graph (NetwokrX) a clique-separator graph
      source (node) soruce node
      subset (node) branch to exclude

    Return: 
        [(soruce, to 1, ..., to final), (soruce, to 1, ..., to final)]
    """
    pathes = []
    path = [source]
    for edge in dfs_edges(graph, source, skip_nodes=skip_nodes):
        if edge[1] & subset:
            continue
        if edge[0] == path[-1]:
            # node of path
            path.append(edge[1])
        else:
            # new path
            if graph.node[path[-1]]['clique']:
                pathes.append(path)
            search_index = 2
            while search_index <= len(path):
                if edge[0] == path[-search_index]:
                    path = path[:-search_index + 1] + [edge[1]]
                    break
                search_index += 1
            else:
                raise Exception("Wrong path structure?", path, edge)
    if graph.node[path[-1]]['clique']:
        pathes.append(path)
    return pathes



def dfs_path2cliques_component(graph, source, component):
    """ Return the DFS paths to cliques from the source.

    Args:
      graph (NetwokrX) a clique-separator graph
      source (node) soruce node
      component (node) number of connected component

    Return: 
        [(soruce, to 1, ..., to final), (soruce, to 1, ..., to final)]
    """
    pathes = []
    path = [source]
    for edge in nx.dfs_edges(graph, source):
        if edge[0] == path[-1]:
            # node of path
            path.append(edge[1])
        else:
            # new path
            pathes.append(path)
            search_index = 2
            while search_index <= len(path):
                if edge[0] == path[-search_index]:
                    path = path[:-search_index + 1] + [edge[1]]
                    break
                search_index += 1
            else:
                raise Exception("Wrong path structure?", path, edge)
    pathes.append(path)
    final_pathes = []
    for p in pathes:
        last_clique = p[-1]
        com = graph.node[last_clique]
        if com['clique'] and com['component'] != component:
            final_pathes.append(p)
            
    return final_pathes

def sample_one_path(dict_of_lists):
    """ Returns a dict of {key: [one item]}

    Args:
       dict_of_lists: a dict of lists
    """
    partitions = []
    for key, pathes in dict_of_lists.items():
        if pathes:
            np.random.shuffle(pathes)
            partitions.append(pathes[0])
    return partitions

    
# def sample_cliques_from_blocks(part_dict):
#     partitions = []
#     for block, pathes in part_dict.items():
#         if pathes:
#             np.random.shuffle(pathes)
#             path = pathes[0]
#             sep = path[0]
#             clq = path[-1]
#             partitions.append((clq, sep))
#     return partitions, 0.0

def sample_cliques_from_blocks(part_dict):
    partitions = []
    for block, pathes in part_dict.items():
        np.random.shuffle(pathes)
        for path in pathes:
            sep = path[0]
            clq = path[-1]
            partitions.append((clq, sep))
    return partitions, 0.0


def sample_cliques_from_components(graph, part):
    if not frozenset([]) in part.keys():
        return part, 0.0
    a = part[frozenset([])]
    components = [graph.node[path[-1]]['component'] for path in a]
    k = collections.Counter(components)
    log_p = np.log(np.array(k.values())).sum()

    components_dict = {c: [] for c in range(graph.num_components)}
    for l in a: 
        components_dict[graph.node[l[-1]]['component']].append(l)
    comp_list = []
    for comp, pt in components_dict.items():
        np.random.shuffle(pt)     
        if pt: 
            comp_list.append(pt[0])
    part[frozenset([])] = comp_list
    return part, log_p


def connect_partition(graph, node):
    """ Returns a list of all connect cliques in the following format

        [(clique, separator), (clique, separator)]
    
    Args:
      graph (NetwokrX) a clique-separator grpah
      node (frozenset) a node
    """
    initial_part = neighboring_nodes(graph, node)
    partition_list, log_p = sample_cliques_from_blocks(initial_part)
    #part, log_p = sample_cliques_from_components(graph, initial_part) 
    return partition_list, log_p



def disconnect_partition(graph, node):
    """ Returns all disconnect cliques in the following format
        
        [(clique, separator), (clique, separator)]
    
    Args:
      graph (NetwokrX) a clique-separator graph
      node (frozenset) a node
    """
    induced_graph = subgraph_induced_by_subset(graph, node)
    leaves = leaf_nodes(induced_graph)
    lf = [(clq, list(induced_graph.predecessors(clq))[0]) for clq in leaves]
    return lf, 0.0




def connect(graph, clique_tupple, node):
    new_clique = clique_tupple[0]
    sep = clique_tupple[2]
    simplix = new_clique - sep - node
    node = list(node)[0]
    # 0 for connection tyep
    edges_to_add = [(node, y) for y in set(simplix)]
    graph.add_edges_from(edges_to_add)


def disconnect_move(graph, clique_tupple, node):
    old_clq = clique_tupple[1]
    sep = clique_tupple[2]
    simplix = old_clq - sep - node
    node = list(node)[0]
    # 1 for disconnect type
    edges_to_remove = [(node, y) for y in set(simplix)]
    graph.remove_edges_from(edges_to_remove)

def organize_into_components(graph, path_list):
    """ Returns a dict of {compnent: pathlist}

    Args:
      graph (NetwokrX) a clique-separator grpah
      path_list (list)  a list of pathes, each ending with a clique
    """
    components = {i: [] for i in range(graph.num_components)}
    for path in path_list: 
        end_node = path[-1]
        end_comp = graph.node[end_node]['component']
        components[end_comp].append(path)
    return components



def to_graph(graph):
    """ Returns the graph underlying the clique-separator graph.

    Returns:
        NetworkX graph
    """

    G = nx.Graph()
    cliques = graph.nodes()
    for c in cliques:
        for n1 in set(c):
            if len(c) == 1:
                G.add_node(n1)
            for n2 in set(c) - set([n1]):
                G.add_edge(n1, n2)
    return G



def dfs_edges(G, source=None, depth_limit=None, skip_nodes = set()):
    """Iterate over edges in a depth-first-search (DFS).

    Perform a depth-first-search over the nodes of `G` and yield
    the edges in order. This may not generate all edges in `G`
    (see `~networkx.algorithms.traversal.edgedfs.edge_dfs`).

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
       Specify starting node for depth-first search and yield edges in
       the component reachable from source.

    depth_limit : int, optional (default=len(G))
       Specify the maximum search depth.

    Yields
    ------
    edge: 2-tuple of nodes
       Yields edges resulting from the depth-first-search.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> list(nx.dfs_edges(G, source=0))
    [(0, 1), (1, 2), (2, 3), (3, 4)]
    >>> list(nx.dfs_edges(G, source=0, depth_limit=2))
    [(0, 1), (1, 2)]

    Notes
    -----
    If a source is not specified then a source is chosen arbitrarily and
    repeatedly until all components in the graph are searched.

    The implementation of this function is adapted from David Eppstein's
    depth-first search function in PADS [1]_, with modifications
    to allow depth limits based on the Wikipedia article
    "Depth-limited search" [2]_.

    See Also
    --------
    dfs_preorder_nodes
    dfs_postorder_nodes
    dfs_labeled_edges
    :func:`~networkx.algorithms.traversal.edgedfs.edge_dfs`
    :func:`~networkx.algorithms.traversal.breadth_first_search.bfs_edges`

    References
    ----------
    .. [1] http://www.ics.uci.edu/~eppstein/PADS
    .. [2] https://en.wikipedia.org/wiki/Depth-limited_search
    """
    if source is None:
        # edges for all components
        nodes = G
    else:
        # edges for components with source
        nodes = [source]

    visited = set()
    if depth_limit is None:
        depth_limit = len(G)
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, depth_limit, iter(G[start]))]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child not in visited and child not in skip_nodes:
                    yield parent, child
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, iter(G[child])))
            except StopIteration:
                stack.pop()


def clique_successors(graph, parents, subset):
    direct_child_clqs = set()
    for p in list(parents): 
        for child in graph.successors(p): 
            if (graph.node[child]['clique']) & (not child & subset): 
                direct_child_clqs.add(child)
    return direct_child_clqs

def separator_predecessors(graph, children, subset):
    direct_parents = set()
    for p in list(children): 
        for child in graph.predecessors(p): 
            if (not graph.node[child]['clique']) & (not child & subset): 
                direct_parents.add(child)
    return direct_parents

