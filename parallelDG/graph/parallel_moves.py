
import networkx as nx
import numpy as np
import random
import scipy.special as sp
from parallelDG.graph import junction_tree as jtlib



def subtree_induced_by_set(tree, node_set):
    """
    This function returns the subtree induced by the set s.

    Args:
       tree (networkx.Graph): A junction tree.
       node_set (set): Subset of the node in the underlying graph of T.

    Returns:
       networkx.Graph: The subtree induced by node_set.
    """
    return tree.subgraph(node_set)


def leaf_nodes(tree):
    """
    This function takes a tree and returns a list of its leaf nodes.

    Args:
        tree (networkx.Graph): The input tree.

    Returns:
        leaf_nodes (list): A list of leaf nodes in the tree.
    """
    # Convert degree view to list of tuples and use list comprehension to find leaf nodes
    return {node for node, degree in dict(tree.degree()).items() if degree == 1}


def neighboring_nodes(tree, subtree):
    """
    This function takes a tree and a subtree, and returns a set of nodes in the tree that are adjacent to at least one node in the subtree but are not in the subtree itself.

    Args:
        tree (networkx.Graph): The input tree.
        subtree (networkx.Graph): The input subtree.

    Returns:
        neighbors (set): A set of nodes in the tree that are adjacent to at least one node in the subtree but are not in the subtree itself.
    """

    # Convert subtree nodes to a set for faster lookup
    subtree_nodes = set(subtree.nodes())

    # Use set comprehension to get all neighbors of nodes in the subtree in the main tree
    all_neighbors = {neighbor for node in subtree_nodes for neighbor in tree.neighbors(node)}

    # Return the difference between all_neighbors and subtree_nodes to get neighbors not in the subtree
    return all_neighbors - subtree_nodes


def propose_connect_moves(tree, subtree):
    """ 
    Proposes a random set of connect moves, given the current state of
    the junction tree. Returns the set of new junction tree nodes,
    after performing the move directly on the tree, and the number
    of those nodes.

    Args:
      tree (networkx.Graph): A junction tree.
      subtree (networkx.Graph): A subtree of the junction tree.

    Returns:
      tuple: The neighboring nodes to the subtree and their count.
    """
    neighboring_nodes_set = find_neighbors_and_adjacent_nodes(tree, subtree)
    return neighboring_nodes_set, len(neighboring_nodes_set)


def propose_disconnect_moves(tree, subtree, all_leafs=False):
    """ 
    Proposes a random set of disconnect moves, given the current state of
    the junction tree. Returns a leaf node of the subtree and its count.

    Args:
      tree (networkx.Graph): A junction tree.
      subtree (networkx.Graph): A subtree of the junction tree.

    Returns:
      tuple: Leaf nodes of the subtree and their count.
    """
   
    leafs = find_leaf_and_adjacent_nodes(subtree)
    if all_leafs:
        return leafs, len(leafs)
    if subtree.order() == 2:
        return [random.choice(list(leafs))], 2
    else:
        return leafs, len(leafs)
    
    

def find_leaf_and_adjacent_nodes(subtree):
    """
    This function returns a list of tuples of leaf nodes and their connected nodes in the tree.

    Args:
       subtree (networkx.Graph): A subtree of the tree.

    Returns:
       list of tuples: Each tuple is in the format (leaf, connected_node) where leaf is a leaf node in the subtree and connected_node is a node in the tree connected to the leaf.
    """
    leaf_nodes_in_subtree = leaf_nodes(subtree)
    leaf_and_adjacent_nodes = []

    for leaf in leaf_nodes_in_subtree:
            adjacent_node = list(subtree.neighbors(leaf))[0]
            leaf_and_adjacent_nodes.append((leaf, adjacent_node))    
    return leaf_and_adjacent_nodes


def find_neighbors_and_adjacent_nodes(tree, subtree):
    """
    This function returns a list of tuples of nodes in the subtree and their connected nodes in the tree.

    Args:
       tree (networkx.Graph): A tree.
       subtree (networkx.Graph): A subtree of the tree.

    Returns:
       list of tuples: Each tuple is in the format (node, connected_node) where node is a node in the subtree and connected_node is a node in the tree connected to the node.
    """
    nodes_and_adjacent_in_tree = []

    for node in subtree.nodes:
        neighboring_nodes = set(tree.neighbors(node)) - set(subtree.neighbors(node))

        for nei_node in neighboring_nodes:
            nodes_and_adjacent_in_tree.append((nei_node, node))
    
    return nodes_and_adjacent_in_tree


def is_adj_clique_in_set(C_adj, move_set): 
    return np.all([C_adj != C[0] for C in move_set])

def reverse_move_leafs_count(current_leafs, Cadj):
    L = len(current_leafs) + 1 * is_adj_clique_in_set(Cadj, current_leafs)
    return L

def reverse_neighbors_count(tree, t_node, num_moves):
    return num_moves + 2 - tree.degree(t_node)



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
