"""
Functions related to junction trees.
"""

import networkx as nx
import numpy as np
import itertools
from functools import reduce
#import parallelDG.graph.decomposable as dlib

class JunctionMap:
    def __init__(self, junction_tree):
        self.graph_nodes = set().union(*junction_tree.nodes())
        self.p = len(self.graph_nodes)
        self.t, self.t2clique = self.create_t_and_t2clique(junction_tree)
        self.node2t = self.create_node2t()


    def create_t_and_t2clique(self, jt, p = None):
        """
        This function transforms a junction tree into a new graph where the original cliques are mapped to integer identifiers.
        Singleton nodes in the original tree, if any, are connected to randomly chosen nodes in the new graph.

        Args:
            jt (networkx.Graph): The input junction tree.

        Returns:
            t (networkx.Graph): The new graph, based on integer identifiers.
            t2clique (dict): A dictionary mapping from new integer identifiers to original cliques.
        """
        # Extract graph nodes

        # Create mappings for junction tree cliques and corresponding integer identifiers
        if not p:
            p = self.p
        t2clique = {k:set() for k in range(p)}

        keys = list(t2clique.keys())
        np.random.shuffle(keys)
        for i, C in enumerate(jt.nodes()):
            k = keys[i]
            t2clique[k] = set(C)         # Convert each frozenset to a set
        clique2t = {frozenset(v): k for k, v in t2clique.items()}  # Use frozenset for dictionary keys

        # Initialize new graph
        t = nx.Graph()
        t.add_nodes_from(range(p))

        # Add edges based on junction tree structure
        edges_to_add = [(clique2t[edge[0]], clique2t[edge[1]]) for edge in jt.edges()]
        t.add_edges_from(edges_to_add)

        # Identify singleton nodes
        singletons = [node for node in t.nodes() if t.degree(node) == 0 and not t2clique[node]]

        # If there are any singleton nodes, randomly connect them to existing nodes
        if singletons:
            non_empty_set = set(range(p)) - set(singletons)
            for node in singletons:
                target_node = np.random.choice(list(non_empty_set))
                t.add_edge(node, target_node)
                # Add singleton nodes to the t2clique dictionary with empty clique
                t2clique[node] = set()
                non_empty_set.add(node)
        return t, t2clique


    def create_node2t(self):
        """
        This function creates a mapping from each node in the original graph to the corresponding nodes in the new graph that include it in their cliques, based on the t2clique mapping.
        """
        node2t = {}
        for t_node, clique in self.t2clique.items():
            for node in clique:
                node2t.setdefault(node, set()).add(t_node)
        return node2t

    def connect(self, t_node, g_node):
        """Add a connection between a graph node and a tree node in the mapping dictionaries.
    
        Args:
            t_node: The tree node identifier.
            g_node: The graph node identifier.
        """
        
        self.node2t[g_node].add(t_node)
        self.t2clique[t_node].add(g_node)


    def disconnect(self, t_node, g_node):
        """Remove a connection between a graph node and a tree node in the mapping dictionaries.

        Args:
            t_node: The tree node identifier.
            g_node: The graph node identifier.
        
        """
        
        self.node2t[g_node].remove(t_node)
        self.t2clique[t_node].remove(g_node)

    def get_cliques(self):
        """ Returns the set of cliques from the t2clique dictionary.

        Returns:
            cliques (list): List of cliques, where each clique is represented as a set of nodes.
        """
        return [frozenset(clique) for clique in self.t2clique.values() if clique]

    def get_separators(self, return_graph_sep = True):
        """ Returns the dictionary of separators from the tree and its t2clique dictionary.

        Returns:
            separators (dict): Dict with separators as keys and list of associated edges as values.
        """
        separators =dict()  # Using a dict to store separators and their associated edges
        t_separators = dict()
        for edge in self.t.edges():
            clique_a = self.t2clique.get(edge[0], set())
            clique_b = self.t2clique.get(edge[1], set())
            #if clique_a and clique_b:  # Ensure both cliques are not empty
            separator = frozenset(clique_a.intersection(clique_b))
            if separator in separators:
                separators[separator].append((clique_a, clique_b))
                t_separators[separator].append(edge)
            else:
                separators[separator] = [(clique_a, clique_b)]
                t_separators[separator] = [edge]
        if return_graph_sep: 
            return separators
        else:
            return t_separators


    def n_subtrees(self, sep):
        if self.t.size() == 0:
            return [1]
        visited = set()
        start_nodes = set()
        leaf = None
        counts = []
        graph_sep = self.t2clique[sep[0]] & self.t2clique[sep[1]]
        for n in self.t.nodes():
            valid_neighs = [ne for ne in self.t.neighbors(n) if graph_sep <= self.t2clique[ne]]
            if len(valid_neighs) == 1 and graph_sep <= self.t2clique[n]:
                leaf = n
                break

        start_nodes.add(leaf)
        prev_visited = 0
        while len(start_nodes) > 0:
            n = start_nodes.pop()
            self.n_subtrees_aux(n, graph_sep, visited, start_nodes)
            counts += [len(visited) - prev_visited]
            prev_visited = len(visited)

        return counts

    def n_subtrees_aux(self, node, graph_sep, visited, start_nodes):
        visited.add(node)
        #for n in nx.neighbors(tree, node):
        for n in self.t.neighbors(node):
            if graph_sep <= self.t2clique[n]:
                if n not in visited:
                    if self.t2clique[n] & self.t2clique[node] == graph_sep:
                        start_nodes.add(n)
                    else:
                        self.n_subtrees_aux(n, graph_sep, visited, start_nodes)

    def log_nu(self, s):
        """ Returns the number of equivalent junction trees for tree where
            tree is cut at the separator s and then constructed again.

        Args:
            s (set): A separator of tree

        Returns:
            float
        """
        f = np.array(self.n_subtrees(s))
        ts = f.ravel().sum()
        ms = len(f) - 1
        return np.log(f).sum() + np.log(ts) * (ms - 1)

    def log_n_junction_trees(self, S):
        """ Returns the number of junction trees equivalent to tree where trees
        is cut as the separators in S. is S i the full set of separators in tree,
        this is the number of junction trees equivalent to tree.

        Args:
            S (list): List of separators of tree
        Returns:
            float
        """
        log_mu = 0.0
        for sep, t_edges in S.items():
            log_mu += self.log_nu(t_edges[0])
        return log_mu

    
    def to_graph(self):
        """
        Returns the graph underlying the integer identifier tree.

        Args:
            t2clique (dict): A dictionary mapping from new integer identifiers to original cliques.

        Returns:
            G (networkx.Graph): An undirected graph.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.p))
        for t_node, clique in self.t2clique.items():
            if len(clique) == 1:
                G.add_node(list(clique)[0])  # add singleton nodes
            else:
                for node1, node2 in itertools.combinations(clique, 2):
                    G.add_edge(node1, node2)
        return G

    def to_junction_tree(self):
        # Creating a mapping from the original nodes to the new nodes (cliques)
        jtree = JunctionTree()
        node_mapping = {frozenset(clique) for node, clique in self.t2clique.items() if clique}
        jtree.add_nodes_from(list(node_mapping))
        edges_to_add = []
        for edge in self.t.edges():
            e0, e1 = frozenset(self.t2clique[edge[0]]), frozenset(self.t2clique[edge[1]])
            if e0 and e1:
                edges_to_add.append((e0, e1))

        jtree.add_edges_from(edges_to_add)
        return jtree

    def randomize_by_jt(self):
        graph = self.to_graph()
        import parallelDG.graph.decomposable as dlib
        jt = dlib.junction_tree(graph)
        randomize(jt)
        self.t, self.t2clique = self.create_t_and_t2clique(jt, jt.order())
        self.node2t = self.create_node2t()
        #self.randomize()
        #self.t, self.t2clique = self.create_t_and_t2clique(jt)
        #self.node2t = self.create_node2t()

    def randomize(self):
        """ Returns a random junction tree equivalent to tree.

        Args:
            s (set): A separator of tree format {sep: [(e1,e2), (e3, e4)]}

        Returns:
            NetworkX graph
        """
        S = self.get_separators(return_graph_sep = False)
        for graph_sep, t_edges in S.items():
            self.randomize_at_sep(graph_sep, t_edges)
        

    def randomize_at_sep(self, graph_sep, t_edges):
        """ Returns a junction tree equivalent to tree where tree is cut at s
        and then reconstructed at random.

        Args:
            s (set): A separator of tree

        Returns:
            NetworkX graph
        """
        ## get teh subtree
        if graph_sep:
            subtree_set = reduce(set.intersection, [self.node2t[n] for n in graph_sep])
            subtree = self.t.subgraph(subtree_set).copy()
        else:
            subtree = self.t.copy()
        subtree.remove_edges_from(t_edges)
        new_edges = random_tree_from_forest(subtree)
        # Remove old edges associated with s
        self.t.remove_edges_from(t_edges)
        # Add the new edges
        self.t.add_edges_from(new_edges)
        

        
    def randomize_bfs(self):
        """ Randomizes the tree"""
        non_empty_cliques = {key for key, it in self.t2clique.items() if it}
        if not non_empty_cliques:  # set a root
            non_empty_cliques = set([np.random.randint(self.p)])
        root = np.random.choice(list(non_empty_cliques),1)[0]
        edges = list(nx.bfs_edges(self.t, root))
        running_set = self.t2clique[edges[0][0]].copy()
        visited_set = {edges[0][0]}
        new_edges = []
        empty_t_nodes = set()
        for edge in edges:
            t_node = edge[1]
            C = self.t2clique[t_node]
            if C: 
                S = running_set & C
                if S:
                    #if S == C:  # C is a subset of S
                    #    self.t2clique[t_node] = set()
                    #    empty_t_nodes.add(t_node)
                    #    continue
                    possible_edges = [v for v in visited_set if S <= self.t2clique[v]]  # generator expression
                    new_link = np.random.choice(possible_edges,1)[0]
                else:
                    new_link = np.random.choice(list(visited_set & non_empty_cliques),1)[0]
                running_set.update(C)
                new_edges.append((new_link, t_node))
                visited_set.add(t_node)
            else:
                empty_t_nodes.add(t_node)
                #new_link = np.random.choice(list(visited_set & non_empty_cliques),1)[0]
        if empty_t_nodes:
            empty_links = np.random.choice(list(non_empty_cliques), len(empty_t_nodes), replace=True)
            new_edges += [(x, y) for x, y in zip(empty_t_nodes, empty_links)]  # Change this line
        # Remove all existing edges from the graph
        self.t.remove_edges_from(list(self.t.edges()))
        # Add the new set of edges to the graph
        self.t.add_edges_from(new_edges)
        #self.node2t = self.create_node2t()

class JunctionTree(nx.Graph):

    def __init__(self, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)
        self.log_nus = {}
        self.separators = None
        self.num_graph_nodes = None
        self.latent = False
        self.clique_hard_threshold = None

    def log_nu(self, sep):
        if sep not in self.log_nus:
            self.log_nus[sep] = log_nu(self, sep)
        return self.log_nus[sep]

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
        return JunctionTree()

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).add_edges_from(ebunch_to_add, **attr)

    def remove_edge(self, u, v):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).remove_edge(u, v)

    def remove_node(self, n):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).remove_node(n)

    def remove_edges_from(self, ebunch):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).remove_edges_from(ebunch)

    def get_separators(self):
        if self.separators is None:
            self.separators = separators(self)
        return self.separators

    def connected_component_vertices(self):
        return [list(c) for c in nx.connected_components(self)]

    def connected_components(self):
        return nx.connected_components(self)

    def log_n_junction_trees(self, seps):
        lm = 0.0
        for sep in seps:
            lm += self.log_nu(sep)
        return lm

    def to_graph(self):
        """ Returns the graph underlying the junction tree.

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


def is_junction_tree(tree):
    """ Checks the junction tree property of a graph.

    Args:
        tree (NetworkX graph): A junction tree

    Returns:
        bool: True if tree is a junction tree
    """
    for n1 in tree.nodes():
        for n2 in tree.nodes():
            if n1 == n2:
                continue
            if n1 <= n2:
                return False

    for n1 in tree.nodes():
        for n2 in tree.nodes():
            if n1 == n2:
                continue
            inter = n1 & n2
            path = nx.shortest_path(tree, source=n1, target=n2)
            for n in path:
                if not inter <= n:
                    return False
    return True


def n_junction_trees_graph(graph):
    """ Returns the number of junction trees of a decomposable graph
    Args: 
      grpah (networkx): a decomposable grpah

        """
    import parallelDG.graph.decomposable as dlib
    seps = dlib.separators(graph)
    jt = dlib.junction_tree(graph)
    return int(np.exp(log_n_junction_trees(jt, seps)))


def n_junction_trees(p):
    """ Returns the number of junction trees with p internal nodes.

    Args:
        p (int): number of internal nodes
    """
    import parallelDG.graph.decomposable as dlib

    graphs = dlib.all_dec_graphs(p)
    num = 0
    for g in graphs:
        seps = dlib.separators(g)
        jt = dlib.junction_tree(g)
        num += int(np.exp(log_n_junction_trees(jt, seps)))
    return num


def subtree_induced_by_subset(tree, s):
    """ Returns the subtree induced by the set s.

    Args:
       tree (NetworkX graph): A junction tree.
       s (set): Subset of the node in the underlying graph of T.
    """
    if len(s) == 0:
        return tree.copy()
    v_prime = {c for c in tree.nodes() if s <= c}
    return tree.subgraph(v_prime).copy()

def induced_subtree_nodes(tree, node, visited, sep):
    neigs = [n for n in tree.neighbors(node)
             if sep <= node and n not in visited]
    visited.add(node)
    if len(neigs) > 0:
        neigs.pop()
        for neig in neigs:
            induced_subtree_nodes(tree, neig, visited, sep)
    return visited


def forest_induced_by_sep(tree, s):
    """ Returns the forest created from the subtree induced by s
    and cut at the separator that equals s.
    This is the forest named F in

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        NetworkX graph: The forest created from the subtree induced by s
    and cut at the separator that equals s.
    """
    F = subtree_induced_by_subset(tree, s)
    edges_to_remove = []
    for e in F.edges():
        if s == e[0] & e[1]:
            edges_to_remove.append(e)
    F.remove_edges_from(edges_to_remove)
    return F


def separators(tree):
    """ Returns a dictionary of separators and corresponding
    edges in the junction tree tree.

    Args:
        tree (NetworkX graph): A junction tree

    Returns:
        dict:  Example {sep1: [sep1_edge1, sep1_edge2, ...], sep2: [...]}
    """
    separators = {}
    for edge in tree.edges():
        sep = edge[0] & edge[1]
        if not sep in separators:
            separators[sep] = set([])
        separators[sep].add(edge)
    return separators

def log_nu_dfs(tree, s):
    """ Returns the number of equivalent junction trees for tree where
        tree is cut at the separator s and then constructed again.

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        float
    """
    
    subtree =  forest_induced_by_sep(tree, s)
    f = [len(x) for x in  nx.connected_components(subtree)]
    ts = np.sum(f)
    ms = len(f) - 1
    return np.log(f).sum() + np.log(ts) * (ms - 1)
    

def log_nu(tree, s):
    """ Returns the number of equivalent junction trees for tree where
        tree is cut at the separator s and then constructed again.

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        float
    """
    f = np.array(n_subtrees(tree, s))
    ts = f.ravel().sum()
    ms = len(f) - 1
    return np.log(f).sum() + np.log(ts) * (ms - 1)


def n_subtrees_aux(tree, node, sep, visited, start_nodes):
    visited.add(node)
    #for n in nx.neighbors(tree, node):
    for n in tree.neighbors(node):
        if sep < n:
            if n not in visited:
                if n & node == sep:
                    start_nodes.add(n)
                else:
                    n_subtrees_aux(tree, n, sep, visited, start_nodes)


def n_subtrees(tree, sep):
    if tree.size() == 0:
        return [1]
    visited = set()
    start_nodes = set()
    leaf = None
    counts = []
    for n in tree.nodes():
        #valid_neighs = [ne for ne in nx.neighbors(tree, n) if sep < ne]
        valid_neighs = [ne for ne in tree.neighbors(n) if sep <= ne]
        if len(valid_neighs) == 1 and sep <= n:
            leaf = n
            break

    start_nodes.add(leaf)
    prev_visited = 0
    while len(start_nodes) > 0:
        n = start_nodes.pop()
        n_subtrees_aux(tree, n, sep, visited, start_nodes)
        counts += [len(visited) - prev_visited]
        prev_visited = len(visited)

    return counts

def log_n_junction_trees_dfs(tree, S):
    """ Returns the number of junction trees equivalent to tree where trees
    is cut as the separators in S. is S i the full set of separators in tree,
    this is the number of junction trees equivalent to tree.

    Args:
        tree (NetworkX graph): A junction tree
        S (list): List of separators of tree

    Returns:
        float
    """
    log_mu = 0.0
    for s in S:
        log_mu += log_nu_dfs(tree, s)
    return log_mu


def log_n_junction_trees(tree, S):
    """ Returns the number of junction trees equivalent to tree where trees
    is cut as the separators in S. is S i the full set of separators in tree,
    this is the number of junction trees equivalent to tree.

    Args:
        tree (NetworkX graph): A junction tree
        S (list): List of separators of tree

    Returns:
        float
    """
    log_mu = 0.0
    for s in S:
        log_mu += log_nu(tree, s)
    return log_mu


def randomize_at_sep(tree, s):
    """ Returns a junction tree equivalent to tree where tree is cut at s
    and then reconstructed at random.

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        NetworkX graph
    """
    F = forest_induced_by_sep(tree, s)
    new_edges = random_tree_from_forest(F)
    # Remove old edges associated with s
    to_remove = []
    for e in tree.edges():  # TODO, get these easier
        if e[0] & e[1] == s:
            to_remove += [(e[0], e[1])]

    tree.remove_edges_from(to_remove)

    # Add the new edges
    tree.add_edges_from(new_edges)
    #for e in new_edges:
    #    tree.add_edge(e[0], e[1])


    
def randomize(tree):
    """ Returns a random junction tree equivalent to tree.

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        NetworkX graph
    """
    S = separators(tree)
    for s in S:
        randomize_at_sep(tree, s)

def random_tree_from_forest(F, edge_label=""):
    """ Returns a random tree from the forest F.

    Args:
        F (NetworkX graph): A forest.
        edge_label (string): Labels for the edges.
    """

    #comps = F.connected_component_vertices()
    comps = [list(c) for c in nx.connected_components(F)]
    #comps = [list(t.nodes()) for t in F.connected_components(prune=False)]
    q = len(comps)
    p = F.order()
    # 1. Label the vertices's
    all_nodes = []
    for i, comp in enumerate(comps):
        for j in range(len(comp)):
            all_nodes.append((i, j))
    # 2. Construct a list v containing q - 2 vertices each chosen at
    #    random with replacement from the set of all p vertices.
    v_ind = np.random.choice(p, size=q-2)

    v = [all_nodes[i] for i in v_ind]
    v_dict = {}
    for (i, j) in v:
        if i not in v_dict:
            v_dict[i] = []
        v_dict[i].append(j)

    # 3. Construct a set w containing q vertices,
    # one chosen at random from each subtree.
    w = []
    for i, c in enumerate(comps):
        # j = np.random.choice(len(c))
        j = np.random.randint(len(c))
        w.append((i, j))

    # 4. Find in w the vertex x with the largest first index that does
    #    not appear as a first index of any vertex in v.
    edges_ind = []
    while not v == []:
        x = None
        #  not in v
        for (i, j) in reversed(w):  # these are ordered
            if i not in v_dict:
                x = (i, j)
                break

        # 5. and 6.
        y = v.pop()  # removes from v
        edges_ind += [(x, y)]
        del v_dict[y[0]][v_dict[y[0]].index(y[1])]  # remove from v_dict
        if v_dict[y[0]] == []:
            v_dict.pop(y[0])
        del w[w.index(x)]  # remove from w_dict

    # 7.
    edges_ind += [(w[0], w[1])]
    edges = [(comps[e[0][0]][e[0][1]], comps[e[1][0]][e[1][1]])
             for e in edges_ind]

    F.add_edges_from(edges, label=edge_label)
    return edges


def graph(tree):
    """ Returns the graph underlying the junction tree.

    Args:
        tree (NetworkX graph): A junction tree

    Returns:
        NetworkX graph
    """
    # if not nx.is_tree(tree):
    #    return tree
        
    G = nx.Graph()
    for c in tree.nodes():
        for n1 in set(c):
            if len(c) == 1:
                G.add_node(n1)
            for n2 in set(c) - set([n1]):
                G.add_edge(n1, n2)
    return G


def peo(tree):
    """ Returns a perfect elimination order and corresponding cliques, separators, histories, , rests for tree.

    Args:
        tree (NetworkX graph): A junction tree.

    Returns:
       tuple: A tuple of form (C, S, H, A, R), where the elemenst are lists of Cliques, Separators, Histories, , Rests, from a perfect elimination order.
    """
    # C = list(nx.dfs_preorder_nodes(tree, tree.nodes()[0])) # nx < 2.x
    C = list(nx.dfs_preorder_nodes(tree, list(tree.nodes)[0])) # nx > 2.x
    S = [set() for j in range(len(C))]
    H = [set() for j in range(len(C))]
    R = [set() for j in range(len(C))]
    A = [set() for j in range(len(C)-1)]
    S[0] = None
    H[0] = C[0]
    R[0] = C[0]
    for j in range(1, len(C)):
        H[j] = H[j-1] | C[j]
        S[j] = H[j-1] & C[j]
        A[j-1] = H[j-1] - S[j]
        R[j] = C[j] - H[j-1]
    return (C, S, H, A, R)


def n_junction_trees_update(new_separators, from_tree, to_tree, log_old_mu):
    """ Returns the new log mu where to_tree has been generated from from_tree2

    Args:
        from_tree (NetworkX graph): A junction tree
        to_tree (NetworkX graph): A junction tree
        new_separators (dict): The separators generated by the CTA.
        log_old_mu: Log of the number of junction trees of from_tree.

    """
    return log_n_junction_trees_update_ratio(new_separators, from_tree, to_tree) + log_old_mu


def log_n_junction_trees_update_ratio(new_separators, from_tree, to_tree):
    """ Returns the log of the ratio of number of junction trees of from_tree and to_tree.

    Args:
        from_tree (NetworkX graph): A junction tree
        to_tree (NetworkX graph): A junction tree
        new_separators (dict): The separators generated by the CTA.
        log_old_mu (float): Log of the number of junction trees of from_tree.

    Returns:
        float: log(mu(to_tree)/mu(from_tree))
    """

    old_full_S = from_tree.get_separators()
    new_full_S = to_tree.get_separators()
    old_subseps = set()
    new_subseps = set()

    # subtract those that has to be "re-calculated"
    for new_s in new_separators:
        for s in old_full_S:
            # the spanning tree for s will be different in the new tree
            # so the old calculation is removed
            if s <= new_s:
                old_subseps.add(s)
    for new_s in new_separators:
        for s in new_full_S:
            if s <= new_s:
                new_subseps.add(s)

    new_partial_mu = to_tree.log_n_junction_trees(new_subseps)
    old_partial_mu = from_tree.log_n_junction_trees(old_subseps)

    return new_partial_mu - old_partial_mu


def to_prufer(tree):
    """ Generate Prufer sequence for tree.

    Args:
        tree (NetwokrX.Graph): a tree.

    Returns:
        list: the Prufer sequence.
    """
    graph = tree.subgraph(tree.nodes())
    if not nx.is_tree(graph):
        return False
    order = graph.order()
    prufer = []
    for _ in range(order-2):
        leafs = [(n, graph.neighbors(n)[0]) for n in graph.nodes() if len(graph.neighbors(n)) == 1]
        leafs.sort()
        prufer.append(leafs[0][1])
        graph.remove_node(leafs[0][0])

    return prufer


def from_prufer(a):
    """
    Prufer sequence to tree
    """
    # n = len(a)
    # T = nx.Graph()
    # T.add_nodes_from(range(1, n+2+1))  # Add extra nodes
    # degree = {n: 0 for n in range(1, n+2+1)}
    # for i in T.nodes():
    #     degree[i] = 1
    # for i in a:
    #     degree[i] += 1
    # for i in a:
    #     for j in T.nodes():
    #         if degree[j] == 1:
    #             T.add_edge(i, j)
    #             degree[i] -= 1
    #             degree[j] -= 1
    #             break
    # print degree
    # u = 0  # last nodes
    # v = 0  # last nodes
    # for i in T.nodes():
    #     if degree[i] == 1:
    #         if u == 0:
    #             u = i
    #         else:
    #             v = i
    #             break
    # T.add_edge(u, v)
    # degree[u] -= 1
    # degree[v] -= 1
    # return T

    n = len(a)
    T = nx.Graph()
    T.add_nodes_from(range(n+2))  # Add extra nodes
    degree = [0 for _ in range(n+2)]
    for i in T.nodes():
        degree[i] = 1
    for i in a:
        degree[i] += 1
    for i in a:
        for j in T.nodes():
            if degree[j] == 1:
                T.add_edge(i, j)
                degree[i] -= 1
                degree[j] -= 1
                break
    u = 0  # last nodes
    v = 0  # last nodes
    for i in T.nodes():
        if degree[i] == 1:
            if u == 0:
                u = i
            else:
                v = i
                break
    T.add_edge(u, v)
    degree[u] -= 1
    degree[v] -= 1
    return T


def jt_to_prufer(tree):
    ind_to_nodes = tree.nodes()
    nodes_to_ind = {ind_to_nodes[i]: i for i in range(tree.order())}
    edges = [(nodes_to_ind[e1], nodes_to_ind[e2]) for (e1, e2) in tree.edges()]
    graph = nx.Graph()
    graph.add_nodes_from(range(tree.order()))
    graph.add_edges_from(edges)



def to_frozenset(G):
    """ Converts a graph with nodes and edges as lists, to a frozenset"""
    g = nx.Graph()

    def _sort(n):
        if hasattr(n, '__iter__'):
            return sorted(n)
        return [n]

    for n1 in G.nodes():
        g.add_node(frozenset(_sort(n1)))

    for n1, n2 in G.edges():
        g.add_edge(frozenset(_sort(n1)), frozenset(_sort(n2)))
    return g
