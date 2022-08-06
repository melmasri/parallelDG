
import networkx as nx
import numpy as np
import random
import scipy.special as sp
from parallelDG.graph import junction_tree as jtlib


def leaf_nodes(tree):
    """ Returns a set of the leaf nodes of tree
    Args:
      tree (NetworkX graph): a tree
    """
    return [x for x in tree.nodes() if tree.degree(x) == 1]



def boundary_cliques_node(tree, node, return_sep=False):
    """ Return boundary cliques for a specific node
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
      cache (set) is used to pass cliques to check against, only used to calculate
                  the probability of the inverse move.
    """
    dumble = False
    T = jtlib.subtree_induced_by_subset(tree, node)
    if len(T) > 1:             # num of nodes is 1
        boundary_cliques = leaf_nodes(T)
    else:
        boundary_cliques = list(T.nodes())

    if len(T) == 2 and len(boundary_cliques) == 2:
        dumble = True
    if return_sep:
        if len(T) > 1:  
            return [(list(T.neighbors(x))[0], x) for x in boundary_cliques], dumble
        else:
            return [(frozenset([]), x) for x in boundary_cliques if x != node], dumble
    else:
        return boundary_cliques, dumble


def neighboring_cliques_node(tree, node):
    """ Return neighboring cliques for the node-induced junction tree
    in  a dictionary. key:item pairs as (connector in node-induced):nei_clique
    Args:
      tree (NetwokrX) a junction tree
      node (frozenset) a node
      empty_node (bool) if empty cliques should be included
    """
    anchor_clq_pairs = list()          # format ((anchor, clq1), (anchor, clq1), ()..) anchor in T, clq not in T
    T = jtlib.subtree_induced_by_subset(tree, node)
    # nei_cliques (connect move)
    for subnode in T:           # subnode is not necessary a boundary clique
        for nei in tree.neighbors(subnode):
            if (not node & nei) and ((node | nei) not in T):
                anchor_clq_pairs.append((subnode, nei))
    # adding single-clique node
    if tree.latent  and len(tree) < tree.clique_hard_threshold and tree.order()==1:
        if node not in T:
            r = np.random.choice(len(T))
            subnode = list(T.nodes())[r]
            anchor_clq_pairs.append((subnode, frozenset()))
    return anchor_clq_pairs



def paritioned_connect_moves(tree, node):
    """ Proposes a random set of connect moves, given the current state of
        the junction tree. Returns the set of new junction tree nodes,
        after performing the move directly on the tree, and the probability
        of those moves.
    Args:
      tree (NetwokrX) a junction tree
      node (frozenset) a node
    """
    part = neighboring_cliques_node(tree, node)
    num = len(part)
    return part, num

def paritioned_disconnect_moves(tree, node):
    """ Proposes a random set of new moves, given the current state of
        the junction tree. Returns the set of new junction tree nodes,
        after performing the move directly on the tree, and the probability
        of those moves.
    Args:
      tree (NetwokrX) a junction tree
      node (frozenset) a node
    """
    
    bd_cliques, dumble = boundary_cliques_node(tree, node, return_sep=True)
    if dumble:
        k = np.random.randint(2)
        return [bd_cliques[k]], 2
    else:
        return bd_cliques, len(bd_cliques)



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
    edges_to_add = []
    if len(new_clique) != 0:     # in case of an empty tree-node
        edges_to_add = [(new_clique, y) for y in tree.neighbors(old_clique)]
        tree.add_node(new_clique)
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


def update_tree_connect(tree, clique):
    """ Updating the junction tree 
        incase of a connect move
    """
    cliques_to_remove = []
    edges_to_add = []
    for n in tree.neighbors(clique):
        if n <= clique:         # connect case
            [edges_to_add.append((y, clique)) for y in tree.neighbors(n)
             if y != clique]
            cliques_to_remove.append(n)
    tree.add_edges_from(edges_to_add)
    tree.remove_nodes_from(cliques_to_remove)

def update_tree_disconnect(tree, clique):
    """ Updating the junction tree 
        incase of a disconnect move
    """
    cliques_to_remove = []
    edges_to_add = []
    if len(clique) == 0:
        return
    for n in tree.neighbors(clique):
        if clique < n:         # connect case
            [edges_to_add.append((y, n)) for y in tree.neighbors(clique)
             if y != n]
            cliques_to_remove.append(clique)
            break               #  this is needed
    tree.add_edges_from(edges_to_add)
    tree.remove_nodes_from(cliques_to_remove)


def update_tree(tree, clique, move_type):
    if move_type == 0:          # connect
        update_tree_connect(tree, clique)
    else:
        update_tree_disconnect(tree, clique)


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
        N = int(np.sum(nei_value_len)) + 1*(len(tree) < tree.clique_hard_threshold)
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
                             index=None,
                             subindex=None):
    new_clique = clique_tupple[0]
    anchor_clique = clique_tupple[2]
    simplix = new_clique - anchor_clique - node
    node = list(node)[0]
    # 0 for connection tyep
    edges_to_add = [(index, subindex, 0, (node, y)) for y in set(simplix)]
    return edges_to_add

def jt_to_graph_disconnect_move(clique_tupple,
                                node,
                                index=None,
                                subindex=None):
    old_clq = clique_tupple[1]
    anchor_clq = clique_tupple[2]
    simplix = old_clq - anchor_clq - node
    node = list(node)[0]
    # 1 for disconnect type
    edges_to_remove = [(index, subindex, 1, (node, y)) for y in set(simplix)]
    return edges_to_remove


# ----------- Green & Thomas move styles
def disconnect_select_subsets(tree, c, node):
    # 2. choose sets
    M = np.random.randint(2, high=len(c)+1)
    N = np.random.randint(1, high=M)
    X = frozenset(np.random.choice(list(c), size=N, replace=False))
    Y = frozenset(np.random.choice(list(c-X), size=M-N, replace=False))
    #print "X: " + str(X)
    #print "Y: " + str(Y)
    return (X, Y)


def disconnect_select_subsets(tree, clique, anchor, node):
    simplix = clique - anchor - node
    # choose from the set of nodes in the simplix
    M = np.random.randint(1, high=len(simplix)+1)
    Y = frozenset(np.random.choice(list(simplix), size=M, replace=False))
    new_clique = clique - Y
    return  clique, new_clique, simplix, Y

def connect_select_subsets(tree,clique, anchor, node):
    # choose from the set of nodes in the simplix
    simplix = clique - anchor - node
    Y = frozenset(np.random.choice(list(simplix), len(simplix), replace=False))
    new_clique = Y | node
    return clique, new_clique, simplix, Y

def connect_logprob(simplix, Y):
    n  = len(simplix)
    k = len(Y)
    return sp.gammaln(n + 1) - sp.gammaln(k + 1) - sp.gammaln(n-k + 1)

def disconnect_logprob(simplix, Y):
    n  = len(simplix)
    k = len(Y)
    return sp.gammaln(n + 1) - sp.gammaln(k + 1) - sp.gammaln(n-k + 1)


def connect_move():
    if Y == simplix: # full connect
        print('Full connect')
        modified_cl = cl | node
        ndlib.connect(jt, cl, modified_cl, anchor_cl)
        ndlib.update_tree(jt, modified_cl, move_type)
    else: 
        intermediary_cl = cl_new
        modified_cl = cl | node
        jt.add_node(intermediary_cl)
        remove_edges = [(cl, anchor_cl)]
        jt.remove_edges_from(remove_edges)
        add_edges  = [(cl, intermediary_cl), (intermediary_cl, anchor_cl)] 
        jt.add_edges_from(add_edges)
        ndlib.connect(jt,cl, modified_cl, anchor_cl)
        ndlib.update_tree(jt, modified_cl, move_type)
    ndlib.update_tree(jt, intermediary_cl, move_type)
