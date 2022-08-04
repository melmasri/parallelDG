import glob
import os

import networkx as nx
import numpy as np
import pandas as pd
from numpy import linalg as la
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from tqdm import tqdm
import random
import datetime
import itertools as its
import copy

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

def plot_heatmap(heatmap, cbar=False, annot=False, xticklabels=1, yticklabels=1):
    mask = np.zeros_like(heatmap)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(heatmap, mask=mask, annot=annot,
                         cmap="Blues",
                         vmin=0.0, vmax=1.0, square=True,
                         cbar=cbar,
                         xticklabels=xticklabels,
                         yticklabels=yticklabels)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
    #sns.set_style("whitegrid")
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=12)


def random_subset(A):
    """ Draws a random subset of elements in a list, inclding the empty set.

    Args:
        A (list)

    Returns:
        set: Subset of A.
    """
    tmp = np.array(list(A))
    bin_samp = np.random.multinomial(1,  [0.5, 0.5], size=len(tmp))
    c = np.ma.masked_array(tmp, mask=bin_samp[:, 0])
    rest = set(c.compressed())
    return rest


def random_element_from_coll(A):
    tmp = np.array(list(A))
    ind = np.random.randint(len(A))
    return tmp[ind]


def l2_loss(m1, m2):
    """ L2 loss between m1 and m2.

    Args:
        m1 (Numpy array): A matrix
        m1 (Numpy array): A matrix

    Returns:
        float
    """
    A = np.matrix(m1)
    B = np.matrix(m2)
    return np.power(la.norm(A - B, "fro"), 2)  # <A-B, A-B>


def l1_loss(m1, m2):
    """ L1 loss.

    Args:
        m1 (Numpy array): A matrix
        m1 (Numpy array): A matrix

    Returns:
        float
    """
    A = np.matrix(m1)
    B = np.matrix(m2)
    p = A.shape[0]
    (sign, logdet) = la.slogdet(A * B.I)
    loss = np.trace(A.transpose() * B.I)  # <A, B.I>
    loss -= sign * logdet
    loss -= p
    return loss


def tpr(true_graph, est_graph):
    """ Calculates the True positive rate of an estimated adjacency matrix.
    """
    N = len(true_graph)
    no_correct = 0.0
    no_false_rej = 0.0

    for i in range(N):
        for j in range(N):
            if est_graph.item(i, j) == 1 and true_graph.item(i, j) == 1:
                    no_correct += 1

            if true_graph.item(i, j) == 1:
                if est_graph.item(i, j) == 0:
                    no_false_rej += 1
    return no_correct / (no_correct + no_false_rej)


def spc1(true_graph, est_graph):
    """
    Takes 2 adjacency matrices.
    """
    N = len(true_graph)
    no_corr_rej = 0.0
    no_wrong_incl = 0.0

    for i in range(N):
        for j in range(N):
            if est_graph.item(i, j) == 1 and true_graph.item(i, j) == 0:
                no_wrong_incl += 1
            if est_graph.item(i, j) == 0 and true_graph.item(i, j) == 0:
                no_corr_rej += 1

    return no_corr_rej / (no_corr_rej + no_wrong_incl)


def get_marg_counts(full_data, subset):
    """ Returns a contingency table in dictionary form.

    Args:
        data (np.array): The data in n x p form.
        subset (list): The subset of interest
    """
    if len(subset) == 0:
        return None
    counts = {}
    for row in full_data:
        cell = tuple(row[subset])
        # print cell
        if cell not in counts:
            counts[cell] = 1
        else:
            counts[cell] += 1
    return counts


def plot_matrix(m, filename, extension, title="Adjmat"):
    """ Plots a 2-dim numpy array as heatmap.
        Args:
            m (numpy array): matrix to plot.
    """
    m1 = np.array(m)
    fig, ax = plt.subplots()
    ax.pcolor(m1, cmap=plt.cm.Blues)
    fig.suptitle(title, fontsize=20)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.savefig(filename+"."+extension, format=extension, dpi=100)


def sample_classification_datasets(mus, covmats, n_samples_in_each_class):
    n_classes = len(covmats)
    n_dim = covmats[0].shape[1]
    # Generate training data
    n_train = [n_samples_in_each_class] * n_classes
    x = np.matrix(np.zeros((sum(n_train), n_dim))).reshape(sum(n_train), n_dim)
    y = np.matrix(np.zeros((sum(n_train), 1), dtype=int)).reshape(sum(n_train), 1)

    for c in range(n_classes):
        fr = sum(n_train[:c])
        to = sum(n_train[:c + 1])
        x[np.ix_(range(fr, to),
                       range(n_dim))] = np.matrix(np.random.multivariate_normal(
                           np.array(mus[c]).flatten(),
                           covmats[c],
                           n_train[c]))
        y[np.ix_(range(fr, to), [0])] = np.matrix(np.ones(n_train[c], dtype=int) * c).T

    ys = pd.Series(np.array(y).flatten(), dtype=int)
    df = pd.DataFrame(x)
    df["y"] = ys
    df = df[["y"] + range(n_dim)]
    df.columns = ["y"] + ["x" + str(i) for i in range(1, n_dim + 1)]
    return df
    # return x, y
    # return pd.DataFrame(x), pd.Series(np.array(y).flatten(), dtype=int)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def gen_prec_mat(graph, a):
    def gen_mat(graph):
        p = graph.order()
        prec_mat = np.zeros(p * p).reshape(p, p)

        adj_mat = nx.to_numpy_array(graph)
        d = np.diag(np.random.uniform(low=a, high=1, size=p))
        for i in range(p - 1):
            for j in range(i + 1, p):
                if adj_mat[i, j] == 1:
                    rn = random.uniform(a, 1)
                    if random.randint(0, 1) == 1:
                        rn *= -1
                    prec_mat[i, j] = rn
                    prec_mat[j, i] = rn

        prec_mat += d
        return prec_mat

    prec_mat = gen_mat(graph)
    p = graph.order()
    i = 0
    e = 0.1
    while not is_pos_def(prec_mat):
        i += 1
        prec_mat += np.diag(np.ones(p)*e)

    return prec_mat


def plot_multiple_traj_statistics(trajs,
                                  burnin_end,
                                  write_to_file=True,
                                  annot=False,
                                  output_directory="./",
                                  file_extension="eps"):
    trajectories = group_trajectories_by_setting(trajs)
    #trajectories = trajs
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    for param_setting, traj_list in trajectories.items():
        #import pdb; pdb.set_trace()
        print("Setting: " + str(traj_list[0].sampling_method['method']))
        print("Average sample time: " + str(np.mean(traj_list[0].time)))

        sns.set_style("whitegrid")
        for t in tqdm(traj_list, total=len(traj_list), desc="Plotting size"):
            t.size(burnin_end).plot()
        if write_to_file:
            plt.savefig(output_directory +"/"+ str(t) + "_size." + file_extension)
        plt.clf()

        sns.set_style("whitegrid")
        for t in tqdm(traj_list, total=len(traj_list), desc="Plotting log-likelihood"):
            t.log_likelihood(burnin_end).plot()
        if write_to_file:
            plt.savefig(output_directory +"/"+ str(t) + "_log-likelihood."+file_extension)
        plt.clf()

        sns.set_style("whitegrid")
        for t in tqdm(traj_list, total=len(traj_list), desc="Plotting autocorr"):
            autocorrelation_plot(t.size(burnin_end))
        if write_to_file:
            plt.savefig(output_directory +"/"+ str(t) + "_autocorr"+"_burnin_"+str(burnin_end)+"."+file_extension)
        plt.clf()


        for i, t in tqdm(enumerate(traj_list), total=len(traj_list),
                         desc="Plotting heatmap, size auto-correlation, MAP and ML graph"):
            plot_heatmap(t.empirical_distribution(burnin_end).heatmap(),
                         xticklabels=np.arange(1, t.seqdist.p +1),
                         yticklabels=np.arange(1, t.seqdist.p +1), annot=annot)
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=6)
            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_heatmap_" + str(i) + "_burnin_"+str(burnin_end)+"."+file_extension)
            plt.clf()

            plot_heatmap(t.empirical_distribution(burnin_end).heatmap(), cbar=True,
                         xticklabels=np.arange(1, t.seqdist.p +1),
                         yticklabels=np.arange(1, t.seqdist.p +1), annot=annot)
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=6)
            cax = plt.gcf().axes[-2]
            cax.tick_params(labelsize=6)

            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_heatmap_cbar_" + str(i) + "_burnin_"+str(burnin_end)+ "."+file_extension)
            plt.clf()

            sns.set_style("white")
            autocorrelation_plot(t.size(burnin_end))
            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_size_autocorr_" + str(i) + "_burnin_"+str(burnin_end)+ "."+file_extension)
            plt.clf()

            top = t.empirical_distribution().mode(1)
            plot_heatmap(nx.to_numpy_array(top[0][0]),
                         xticklabels=np.arange(1, t.seqdist.p +1),
                         yticklabels=np.arange(1, t.seqdist.p +1)
                         )
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=6)
            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_map_" + str(i) + "."+file_extension)
            plt.clf()

            plot_heatmap(nx.to_numpy_array(t.maximum_likelihood_graph()),
                         xticklabels=np.arange(1, t.seqdist.p +1),
                         yticklabels=np.arange(1, t.seqdist.p +1)
                         )
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=6)
            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_ml_" + str(i) + "."+file_extension)
            plt.clf()

def read_all_trajectories_in_dir(directory):

    from parallelDG.graph import trajectory as gtraj
    trajlist = []

    for filename in glob.glob(directory + "/*.json"):
        print("Loading: " + str(filename))
        t = gtraj.Trajectory()
        t.read_file(filename)
        trajlist.append(t)

    return trajlist

def group_trajectories_by_setting(trajlist):
    # Gather all with the same parameter setting
    trajectories = {}
    for t in trajlist:
        if str(t) not in trajectories:
            trajectories[str(t)] = []
        trajectories[str(t)].append(t)

    return trajectories


def plot_graph_traj_statistics(graph_traj, write_to_file=False):
    top = graph_traj.empirical_distribution().mode(5)
    print("Probability\tEdge list: ")
    for graph, prob in top:
        print(str(prob) + "\t\t" + str(list(graph.edges())))

    graph_traj.size().plot()
    if write_to_file:
        plt.savefig(str(graph_traj)+"_size.png")
    plt.clf()

    autocorrelation_plot(graph_traj.size())
    if write_to_file:
        plt.savefig(str(graph_traj)+"_autocorr.png")
    plt.clf()

    graph_traj.log_likelihood().plot()
    if write_to_file:
        plt.savefig(str(graph_traj)+"_loglik.png")
    plt.clf()

    plot_heatmap(graph_traj.empirical_distribution().heatmap())
    if write_to_file:
        plt.savefig(str(graph_traj)+"_heatmap.png")
    plt.clf()

    top = graph_traj.empirical_distribution().mode(1)
    plot_heatmap(nx.to_numpy_array(top[0][0]))
    if write_to_file:
        plt.savefig(str(graph_traj)+"_map.png")
    plt.clf()

    nx.draw(graph_traj.maximum_likelihood_graph(), with_labels=True)
    if write_to_file:
        plt.savefig(str(graph_traj)+"_MAP.png")
    plt.clf()



def write_traj_to_file(graph_trajectory,
                       dirt,
                       output_filename=None,
                       output_format=None):
    date = datetime.datetime.today().strftime('%Y%m%d%H%m%S')
    if not os.path.exists(dirt):
        os.makedirs(dirt)
    if output_format == 'benchpress':
        filename = dirt + "/"+output_filename
        df = graph_trajectory.graph_diff_trajectory_df()
        df.to_csv(filename, sep=",", index=False)
    else:
        filename = dirt + "/" + str(graph_trajectory) + "_" + date + ".json"
        graph_trajectory.write_file(filename=filename)

    print("wrote file: " + filename)
 


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return its.chain.from_iterable(its.combinations(s, r) for r in range(len(s)+1))

def edges_powerset(p):
    "return all graphs of size p"
    return list(its.combinations(list(range(p)), 2))


def hash_graph(G):
    "hashing a graph by ordering the edge set"
    ed = [tuple(sorted(e)) for e in G.edges()]
    return hash(str(sorted(ed)))

def most_sampled_graph(traj):
    """ Returns the most sampled graph from a graph trajectories"""
    graph_dict = {}
    for g in traj:
        if not g:
            continue
        b = hash_graph(g)
        if b in graph_dict:
            graph_dict[b]['count'] += 1.0
        else:
            graph_dict[b] = {'count': 1.0, 'g': g}

    key = max(graph_dict, key=lambda k: graph_dict[k]['count'])
    return graph_dict[key]


def count_decomposable_graphs(size, traj, decomp_index_table):
    "counting the number of decomposable graphs"
    p = size
    graph_dict = copy.deepcopy(decomp_index_table)
    prev_hash = None # to remove repeated graph, where no traj changes
    for g in traj:
        if not g:
            continue
        b = hash_graph(g)
        if b == prev_hash:
            continue
        prev_hash = b
        c = graph_dict[b]['count']
        graph_dict[b]['count'] = 1 + c
                                
    return graph_dict


def pdf_plot(x):
    "plotting the probability density function"
    a = float(np.sum(x))
    b = np.cumsum(x)
    ax = pd.Series(b/a).plot()
    plt.plot([0, 1], [0, 1], transform=ax.transAxes)


def acceptance_ratio(traj):
    x = map(hash_graph, traj)
    # [hash_graph(g) for g in traj]
    dx = (np.diff(x)!=0) * 1.0
    ar = np.mean(dx, dtype=float)
    par = np.cumsum(dx, dtype=float)/(np.array(range(len(dx))) + 1.0)
    print('Acceptance ratio {:2f}'.format(ar))
    return ar, par

def autocorrelation_plot(series,
                         lower=None,
                         upper=None,
                         lag=None,
                         ax=None, **kwds):
    # require scipy
    import scipy
    import matplotlib.pyplot as plt

    return_series = kwds.get('return_series', False)
    n = len(series)
    data = np.asarray(series)

    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
    if lag:
        x = np.arange(lag) + 1
    else:
        x = np.arange(n) + 1
    y = [r(loc) for loc in x]
    if return_series:
        return y
    if ax is None:
        ax = plt.gca(xlim=(1, np.max(x)), ylim=(-1.0, 1.0))
    # customize the z's
    if lower:
        z95, z99 = scipy.stats.t.ppf((1 + np.array([lower,upper])) / 2., 1e9)

        ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="grey")
        ax.axhline(y=z95 / np.sqrt(n), color="grey")
        ax.axhline(y=-z95 / np.sqrt(n), color="grey")
        ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="grey")
    
    ax.axhline(y=0.0, color="grey")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if "label" in kwds:
        ax.legend()
    ax.grid()
    return ax

    
def filter_decomposable_graphs(graphs, graph_size):
    """ returns decomposable graphs out of powerset of graphs 
        and numer of junction trees per graph
    """
    decomp = dict()
    import parallelDG.graph.junction_tree as jtlib
    for a in graphs:
        a = list(a)
        G = nx.Graph()
        G.add_nodes_from(range(graph_size))
        G.add_edges_from(a)
        if nx.is_chordal(G): 
            b = hash_graph(G)
            n_jt = jtlib.n_junction_trees_graph(G)
            decomp[b] = {'count': 0, 'nedges': G.size(),
                         'g': a,
                         'n_jt':
                         n_jt, 'nnodes': graph_size}
        G.clear()
    return decomp

