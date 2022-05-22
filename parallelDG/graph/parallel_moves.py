
import networkx as nx
import numpy as np
import random
import scipy.special as sp
from parallelDG.graph import junction_tree as jtlib


def leaf_nodes(tree, return_edges=False):
    """ Returns a set of the leaf nodes of tree
    Args:
      tree (NetworkX graph): a tree
    """
    if not return_edges:
        return {x for x in tree.nodes() if tree.degree(x) == 1}
    else:
        return {x: list(tree.neighbors(x))[0] for x in tree.nodes() if tree.degree(x) == 1}


def boundary_cliques_node(tree, node, ret_edges=False):
    """ Return boundary cliques for a specific node
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
      cache (set) is used to pass cliques to check against, only used to calculate
                  the probability of the inverse move.
    """
    # boundary cliques (disconnect move)
    boundary_cliques = set()
    dumble = False
    T = jtlib.subtree_induced_by_subset(tree, node)
    if len(T) > 1:             # num of nodes is 1
        boundary_cliques = leaf_nodes(T)
    else:
        boundary_cliques = set(list(T.nodes()))

    if len(T) == 2 and len(boundary_cliques) == 2:
        dumble = True
    if ret_edges:
        if len(T)>1:   
            return {x: list(T.neighbors(x))[0] for x in boundary_cliques}, dumble
        else:
            return {x: frozenset([]) for x in boundary_cliques}, dumble
    else:
        return boundary_cliques, dumble


def neighboring_cliques_node(tree, node, empty_node=False):
    """ Return neighboring cliques for the node-induced junction tree
    in  a dictionary. key:item pairs as (connector in node-induced):nei_clique
    Args:
      tree (NetwokrX) a junction tree
      node (frozenset) a node
      empty_node (bool) if empty cliques should be included
    """
    nei_cliques = dict()
    T = jtlib.subtree_induced_by_subset(tree, node)
    # nei_cliques (connect move)
    for subnode in T:           # subnode is not necessary a boundary clique
        for nei in tree.neighbors(subnode):
            if (not node & nei) and ((node | nei) not in T):
                if subnode in nei_cliques.keys():
                    nei_cliques[subnode].append(nei)
                else:
                    nei_cliques[subnode] = [nei]
    # adding single-clique node
    if tree.latent and empty_node and len(tree) < 2*tree.num_graph_nodes:
        if node not in T:
            r = np.random.choice(len(T))
            subnode = list(T.nodes())[r]
            if subnode in nei_cliques.keys():
                nei_cliques[subnode].append(frozenset())
            else:
                nei_cliques[subnode] = [frozenset()]
    return nei_cliques


def propose_connect_moves(tree, node):
    """ Proposes a random set of connect moves, given the current state of
        the junction tree. Returns the set of new junction tree nodes,
        after performing the move directly on the tree, and the probability
        of those moves.
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
    """
    if not type(node) is set and not type(node) is frozenset:
        node = frozenset([node])
    
    nei_cliques = neighboring_cliques_node(tree, node)
    if not nei_cliques:
        return [None] * 4
    nei_value_len = [len(x) for x in nei_cliques.values()]
    N = int(np.sum(nei_value_len))
    k = np.random.randint(N) + 1
    #k = 1
    nei_n = np.random.choice(N, k, replace=False).tolist()
    new_cliques = set()
    if N > 0:
        keys = nei_cliques.keys()
        aux = list(range(len(keys)))
        index = np.repeat(aux, nei_value_len).tolist()
        a = [index[i] for i in nei_n]
        values, counts = np.unique(a, return_counts=True)
        for i in range(len(values)):
            conn = keys[values[i]]
            j = counts[i]
            nei = nei_cliques[conn]
            np.random.shuffle(nei)
            for old_node in nei[:j]:
                X = node | old_node
                connect(tree, old_node, X, conn)
                new_cliques.add(X)
        if k>N:
            import pdb; pdb.set_trace()
    return new_cliques, log_prob(N, k, 1), N, k


def paritioned_connect_moves(tree, node, empty_node=True):
    """ Proposes a random set of connect moves, given the current state of
        the junction tree. Returns the set of new junction tree nodes,
        after performing the move directly on the tree, and the probability
        of those moves.
    Args:
      tree (NetwokrX) a junction tree
      node (frozenset) a node
    """
    return neighboring_cliques_node(tree, node, empty_node=empty_node)

def paritioned_disconnect_moves(tree, node):
    """ Proposes a random set of new moves, given the current state of
        the junction tree. Returns the set of new junction tree nodes,
        after performing the move directly on the tree, and the probability
        of those moves.
    Args:
      tree (NetwokrX) a junction tree
      node (frozenset) a node
    """
    
    bd_cliques, dumble = boundary_cliques_node(tree, node, ret_edges=True)
    if dumble:
        k = np.random.randint(2)
        ky = bd_cliques.keys()
        return {ky[k]: bd_cliques.pop(ky[k])}
    else:
        return bd_cliques



def propose_disconnect_moves(tree, node, *cache):
    """ Proposes a random set of new moves, given the current state of
        the junction tree. Returns the set of new junction tree nodes,
        after performing the move directly on the tree, and the probability
        of those moves.
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
    """
    if not type(node) is set and not type(node) is frozenset:
        node = frozenset([node])

    bd_cliques, dumble = boundary_cliques_node(tree, node, *cache)
    if not bd_cliques:
        return [None] * 4
    N = len(bd_cliques)
    if dumble:
        k = 1
        N = 2
    else:
        k = np.random.randint(N) + 1
    subset = np.random.choice(N, k, replace=False).tolist()
    new_cliques = set()
    if N > 0:
        bb = list(bd_cliques)
        for i in subset:
            old_node = bb[i]
            X = old_node - node
            disconnect(tree, old_node, X)
            new_cliques.add(X)
        if k>N:
            import pdb; pdb.set_trace()
    return new_cliques, log_prob(N, k, 1), N, k


def disconnect(tree, old_clique, new_clique):
    if len(new_clique) != 0:     # in case of an empty tree-node
        edges_to_add = [(new_clique, y) for y in tree.neighbors(old_clique)]
        tree.add_node(new_clique)
        tree.add_edges_from(edges_to_add)
        tree.remove_node(old_clique)
    else:
        if tree.degree(old_clique) != 1:
            # select a maximal clique
            for nei in tree.neighbors(old_clique):
                if old_clique < nei:
                    edges_to_add = [(nei, y)
                                    for y in tree.neighbors(old_clique)
                                    if y != nei]
                    break
            tree.add_edges_from(edges_to_add)
        tree.remove_node(old_clique)
    if not tree.latent:
        update_tree(tree, new_clique)


def connect(tree, old_clique, new_clique, anchor_clique=None):
    tree.add_node(new_clique)
    # import pdb; pdb.set_trace()
    if old_clique:  # not an empty clique-node
        edges_to_add = [(new_clique, y) for y in tree.neighbors(old_clique)
                        if y != new_clique]
        tree.remove_node(old_clique)
        tree.add_edges_from(edges_to_add)
    else:  # empty clique-node
        edges_to_add = [(new_clique, anchor_clique)]
        tree.add_edges_from(edges_to_add)
    if not tree.latent:
        update_tree(tree, new_clique)


def update_tree(tree, clique):
    cliques_to_remove = []
    edges_to_add = []
    for n in tree.neighbors(clique):
        if n <= clique:
            [edges_to_add.append((y, clique)) for y in tree.neighbors(n)
             if y != clique]
            cliques_to_remove.append(n)
        if clique < n:
            [edges_to_add.append((y, n)) for y in tree.neighbors(clique)
             if y != n]
            cliques_to_remove.append(n) 
            tree.add_edges_from(edges_to_add)
    tree.add_edges_from(edges_to_add)
    tree.remove_nodes_from(cliques_to_remove)


def log_prob(n, k, m=0):
    """ returns the log probability of choosing k out of n
    Args:
    n (integer)
    k (interger) <= n
    m (integer) nu
    """
    return - np.log(sp.binom(n, k)) - m*np.log(2)       # np.sum(np.log(m))




def inverse_proposal_prob(tree, node, new_cliques, move_type):
    """ Returns the log probability of the inverse propoal"""
    if not type(node) is frozenset:
        node = frozenset([node])
    k = len(new_cliques)
    if move_type == 0:               # move_type ==0 disconnect
        bd_cliques, dumble = boundary_cliques_node(tree, node, new_cliques)
        N = len(bd_cliques)
        if dumble:
            N = 2
    else:                       # inverse is connect
        nei_cliques = neighboring_cliques_node(tree, node, False)
        nei_value_len = [len(x) for x in nei_cliques.values()]
        N = int(np.sum(nei_value_len)) + 1*(len(tree) < tree.num_graph_nodes)
    if N < k:
        import pdb;pdb.set_trace()
    return log_prob(N, k, 1), N, k


def revert_moves(tree, node, cliques):
    """ Revert moves in the junction tree
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
      cliques (dict or set) of cliques to revert, diconnect if node in cliques
              otherwise connect.
    """
    # TODO: use type(cliques)=dict() to distinguish betwen cliques
    if not cliques:
        return None
    
    if not type(node) is frozenset:
        node = frozenset([node])

    for nd in cliques:
        if node & nd:           # disconnect
            X = nd - node
            disconnect(tree, nd, X)
        else:       # connect move
            if nd:  # not an empty node
                X = node | nd
                connect(tree, nd, X)
            else:               # empty node
                X = node | nd
                T = jtlib.subtree_induced_by_subset(tree, node)
                conn = list(T.nodes() - cliques)[0]
                connect(tree, nd, X, conn)


def jt_to_graph_connect_move(clique_tupple,
                             node,
                             i=None):
    new_clique = clique_tupple[0]
    simplix = new_clique - node
    node = list(node)[0]
    # 0 for connection tyep
    edges_to_add = [(i, 0, (node,y)) for y in set(simplix)]
    return edges_to_add

def jt_to_graph_disconnect_move(clique_tupple,
                                node,
                                i=None):
    old_clq = clique_tupple[1]
    anchor_clq = clique_tupple[2]
    simplix = old_clq - anchor_clq
    node = list(node)[0]
    # 1 for disconnect type
    edges_to_remove = [(i, 1, (node, y)) for y in set(simplix)]
    return edges_to_remove



def is_isomorphic(jt_traj, graph_traj, graph_updates):
    """ Test isomphisim between graph and jt trajectories"""
    indx = [x[5] for x in graph_updates]
    for k in np.where(np.diff(indx) != 0)[0]:
        x = graph_updates[k]
        j = x[5]
        if not nx.is_isomorphic(jtlib.graph(jt_traj[j]), graph_traj[k+1]):
            print("{}, {}".format(k, j))
            break
