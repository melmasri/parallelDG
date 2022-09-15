"""
Metropolis-Hastings Parallel clique sampler
"""
from multiprocessing import Process
import multiprocessing

import time
import datetime
import os
import numpy as np
from tqdm import tqdm

import parallelDG.distributions.sequential_junction_tree_distributions as seqdist
import parallelDG.graph.graph as glib
import parallelDG.graph.junction_tree as jtlib
import parallelDG.graph.trajectory as mcmctraj
import parallelDG.graph.decomposable as dlib
import parallelDG.graph.clique_separator_graph as clib
import networkx as nx
import parallelDG.auxiliary_functions as aux

import copy
def sample_trajectory(n_samples,
                      randomize,
                      sd,
                      sd_graph,
                      init_graph=None,
                      reset_cache=True,
                      **args):
    seed = args.get('seed', int(time.time()))
    np.random.seed(seed)
    if init_graph:
        graph = init_graph
        jt = dlib.junction_tree(graph)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(range(sd.p))
        dummy_tree = jtlib.to_frozenset(nx.random_tree(n=sd.p, seed=seed))
        jt = jtlib.JunctionTree()
        jt.add_nodes_from(dummy_tree.nodes())
        jt.add_edges_from(dummy_tree.edges())
        jt.num_graph_nodes = sd.p


    cs = clib.clique_separator_graph(jt)
    if reset_cache is True:
        sd.cache = {}

    graph_traj = [None] * n_samples
    graph_traj[0] = graph.copy()
    log_prob_traj = [None] * n_samples

    gtraj = mcmctraj.Trajectory()
    gtraj.trajectory_type = "clique_separator_graph"
    gtraj.set_sampling_method({"method": "mh",
                               "params": {"samples": n_samples}
                               })
    gtraj.set_sequential_distribution(sd)
    gtraj.set_graph_prior(sd_graph)
    gtraj.set_init_graph(graph) 
    gtraj.set_init_jt(cs)

    log_prob_traj[0] = sd.log_likelihood(jt)
    update_moves = list()
    num_nodes = len(graph)
    k = int(0)
    acc_ratios = []
    tic = time.time()

    for i in tqdm(range(1, n_samples), desc="Parallel Metropolis-Hastings samples"):
        cs = clib.clique_separator_graph(dlib.junction_tree(graph))
        #import pdb; pdb.set_trace()
        node = frozenset([np.random.randint(num_nodes)])
        move_type = np.random.randint(2)
        log_p = 0.0
        if move_type == 0:          # connect
            partitions, log_q = clib.connect_partition(cs, node)
            log_q = 0.0
            np.random.shuffle(partitions)             #  shuffling the output
            if not partitions:
                log_prob_traj[i] = log_prob_traj[i-1]
                continue
            for clq, sep in partitions:
                #print((i,move_type, clq, sep, node))
                clq_prop = clq | node
                sep_prop = sep | node
                # New clique log-lik + prior
                log_p2 = sd.log_likelihood_partial(
                    cliques=[clq_prop],
                    separators={sep_prop: set([(sep_prop, clq_prop)])})
                log_g2 = sd_graph.log_prior_partial(clq_prop, sep_prop)
                # old clique log-like + prior
                log_p1 = sd.log_likelihood_partial(
                    [clq],
                    {sep: set([(sep, clq)])})
                log_g1 = sd_graph.log_prior_partial(clq, sep)
                log_g1 = log_g2 = 0
                acc_ratios.append((i, log_p2 - log_p1, log_g2 - log_g1))
                alpha = min(1, np.exp(log_p2 + log_g2 - log_p1 - log_g1))
                k += int(1)
                if np.random.uniform() <= alpha:
                    clique_tuple = (clq_prop, clq, sep)
                    clib.connect(graph, clique_tuple, node)
                    if not nx.is_chordal(graph):
                        import pdb; pdb.set_trace()
                    log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                    update_moves.append((i, k,
                                         move_type,
                                         node,
                                         clique_tuple))
        else:                       # diconnect
            partitions, log_q = clib.disconnect_partition(cs, node)
            np.random.shuffle(partitions)
            if not partitions:
                log_prob_traj[i] = log_prob_traj[i-1]
                continue
            for clq, sep in partitions:
                #print((i,move_type, clq, sep, node))
                clq_prop = clq - node
                sep_prop = sep - node
                # New clique log-lik + prior
                log_p2 = sd.log_likelihood_partial(
                    [clq_prop],
                    {sep_prop: set([(sep_prop, clq_prop)])})
                log_g2 = sd_graph.log_prior_partial(clq_prop, sep_prop)
                # old clique + prior
                log_p1 = sd.log_likelihood_partial([clq],
                                                   {sep: set([(sep, clq)])})
                log_g1 = sd_graph.log_prior_partial(clq, sep)
                log_g1 = log_g2 = 0
                acc_ratios.append((i, log_p2 - log_p1, log_g2 - log_g1))
                alpha = min(1, np.exp(log_p2 + log_g2 - log_p1 - log_g1))
                k += int(1)
                if np.random.uniform() <= alpha:
                    clique_tuple = (clq_prop, clq, sep)
                    clib.disconnect_move(graph, clique_tuple, node)
                    if not nx.is_chordal(graph):
                        import pdb; pdb.set_trace()
                    log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                    update_moves.append((i, k,
                                         move_type,
                                         node,
                                         clique_tuple))
        log_prob_traj[i] = log_prob_traj[i-1] + log_p
        graph_traj[i] = graph.copy()


    toc = time.time()
    gtraj.set_logl(log_prob_traj)
    gtraj.set_nupdates(k)
    gtraj.set_time(toc-tic)
    gtraj.set_jt_updates(update_moves)
    # gtraj.jt_trajectory = jt_traj
    gtraj.trajectory = graph_traj
    gtraj.dummy = acc_ratios
    print('Total of {} updates, for an average of {:.2f} per iteration or {:.2f}updates/sec'.format(k, float(k)/n_samples,k/(toc-tic)))
    print('Acceptance rate {:.4f}'.format(len(update_moves)/float(k)))
    return gtraj


def get_prior(graph_prior):
    sd = None
    if graph_prior[0] == "mbc":
        if len(graph_prior) > 1:
            alpha = float(graph_prior[1])
            beta = float(graph_prior[2])
        else:
            alpha = 2.0
            beta = 4.0
        sd = seqdist.ModifiedBornnCaron(alpha, beta)
    if graph_prior[0] == "edgepenalty":
        if len(graph_prior) > 1:
            alpha = float(graph_prior[1])
        else:
            alpha = 0.001
        sd = seqdist.EdgePenalty(alpha)
    if graph_prior[0] == "uniform":
        sd = seqdist.GraphUniform()
    # default prior
    if not sd:
        sd = get_prior(["mbc", 2.0, 4.0])
    return sd


def sample_trajectory_uniform(
        n_samples,
        randomize=100,
        graph_prior=['uniform'],
        graph_size=10,
        cache={}, **args):
    sd = seqdist.UniformJTDistribution(graph_size)
    sd_graph = get_prior(graph_prior)

    latent = True
    if 'latent' in args:
        latent = args.get('latent')

    if latent:
        sim = sample_trajectory(n_samples=n_samples,
                                randomize=randomize,
                                sd=sd,
                                sd_graph=sd_graph,
                                **args)
    else:
        sim = sample_trajectory_single_move(n_samples=n_samples,
                                            randomize=randomize,
                                            sd=sd,
                                            sd_graph=sd_graph,
                                            **args)
    return sim



def sample_trajectory_ggm(dataframe,
                          n_samples,
                          randomize=100,
                          D=None,
                          delta=1.0,
                          graph_prior=['mbc', 2.0, 4.0],
                          cache={}, **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, cache)
    sd_graph = get_prior(graph_prior)

    if 'single_move' in args:
        traj = sample_trajectory_single_move(n_samples,
                                             randomize, sd,
                                             sd_graph,
                                             **args)
    else:
        traj = sample_trajectory(n_samples,
                                 randomize,
                                 sd,
                                 sd_graph,
                                 **args)
    return traj


def trajectory_to_file(n_samples,
                       randomize,
                       seqdist,
                       seqdist_graph,
                       reset_cache=True,
                       output_directory=".",
                       reseed=False,
                       **args):
    """ Writes the trajectory of graphs generated by particle Gibbs to file.

    Args:
        n_samples (int): Number of Gibbs iterations (samples)
        seq_dist (SequentialJTDistributions): the distribution to be sampled from
        filename_prefix (string): prefix to the filename

    Returns:
        mcmctraj.Trajectory: Markov chain of underlying graphs of the junction trees sampled.

    """
    if reseed is True:
        np.random.seed()

    #print (n_particles, alpha, beta, radius, n_samples, str(seqdist), reset_cache)
    graph_trajectory = sample_trajectory(n_samples,
                                         randomize,
                                         seqdist,
                                         seqdist_graph,
                                         reset_cache=reset_cache,
                                         **args)
    output_filename = args.get("output_filename", None)
    seed = args.get('seed', int(time.time()))
    if not output_filename:
        output_filename = 'graph_traj_'+str(seed)+'.csv'
    output_format = args.get("output_format", None)
    aux.write_traj_to_file(graph_trajectory,
                           dirt=output_directory,
                           output_filename=output_filename,
                           output_format=output_format
    )
    return graph_trajectory


def sample_trajectories_ggm_to_file(dataframe,
                                    n_samples,
                                    randomize=[100],
                                    delta=[1.0],
                                    D=None,
                                    reset_cache=True,
                                    reps=1,
                                    graph_prior=['mbc', 2.0, 4.0],
                                    output_directory=".",
                                    **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd_graph = get_prior(graph_prior)
    graph_trajectories = []
    for _ in range(reps):
        for T in n_samples:
            for r in randomize:
                for d in delta:
                    sd = seqdist.GGMJTPosterior()
                    sd.init_model(np.asmatrix(dataframe), D, d, {})
                    graph_trajectory = trajectory_to_file(n_samples=T,
                                                          randomize=r,
                                                          seqdist=sd,
                                                          seqdist_graph=sd_graph,
                                                          reset_cache=reset_cache,
                                                          output_directory=output_directory,
                                                          **args)
                    graph_trajectories.append(graph_trajectory)
    return graph_trajectories


def sample_trajectories_ggm_parallel(dataframe,
                                     n_samples,
                                     randomize=[100],
                                     D=None,
                                     delta=[1.0,],
                                     reset_cache=True,
                                     reps=1,
                                     graph_prior=['mbc', 2.0, 4.0],
                                     **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd_graph = get_prior(graph_prior)
    queue = multiprocessing.Queue()
    processes = []
    rets = []
    seed = args.get(seed,None)
    for _ in range(reps):
        for T in n_samples:
            for r in randomize:
                for d in delta:
                    sd = seqdist.GGMJTPosterior()
                    sd.init_model(np.asmatrix(dataframe), D, d, {})
                    print("Starting: " + str((T, r, str(sd), reset_cache, True)))
                    proc = Process(target=trajectory_to_queue,
                                   args=(T, r, sd, sd_graph,
                                         queue, reset_cache, True))
                    processes.append(proc)
                    proc.start()
                    time.sleep(2)

    for _ in processes:
        ret = queue.get() # will block
        rets.append(ret)
    for p in processes:
        p.join()

    output_directory = './'
    output_filename = output_format = None
    if "output_directory" in args:
        output_directory = args["output_directory"]
    if "output_filename" in args:
        output_filename = args["output_filename"]
    if "output_format" in args:
        output_format = args["output_format"]

    for traj in rets:
        aux.write_traj_to_file(traj,
                               dirt=output_directory,
                               output_filename=output_filename,
                               output_format=output_format
                               )
    return rets



def sample_trajectory_loglin(dataframe,
                             n_samples,
                             randomize,
                             pseudo_obs=1.0,
                             graph_prior=['mbc', 2.0, 4.0],
                             reset_cache=True, **args):
    p = dataframe.shape[1]
    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])

    sd = seqdist.LogLinearJTPosterior()
    sd.init_model(dataframe.to_numpy(), pseudo_obs, levels, {})
    sd_graph = get_prior(graph_prior)
    return sample_trajectory(n_samples,
                             randomize,
                             sd,
                             sd_graph,
                             reset_cache=reset_cache,
                             **args)


def sample_trajectories_loglin_to_file(dataframe,
                                       n_samples,
                                       randomize=[100, ],
                                       pseudo_obs=[1.0, ],
                                       reset_cache=True,
                                       reps=1,
                                       graph_prior=['mbc', 2.0, 4.0],
                                       output_directory=".",
                                       **args):
    p = dataframe.shape[1]
    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])
    dt = dataframe.to_numpy()
    sd_graph = get_prior(graph_prior)
    graph_trajectories = []
    for _ in range(reps):
        for T in n_samples:
            for r in randomize:
                for p in pseudo_obs:
                    sd = seqdist.LogLinearJTPosterior()
                    sd.init_model(dt, p, levels, {})   
                    graph_trajectory = trajectory_to_file(n_samples=T,
                                                          randomize=r,
                                                          seqdist=sd,
                                                          seqdist_graph=sd_graph,
                                                          reset_cache=reset_cache,
                                                          output_directory=output_directory,
                                                          **args)
                    graph_trajectories.append(graph_trajectory)
    return graph_trajectories

def sample_trajectories_loglin_parallel(dataframe,
                                        n_samples,
                                        randomize=[100, ],
                                        pseudo_obs=[1.0, ],
                                        reset_cache=True,
                                        reps=1,
                                        graph_prior=['mbc', 2.0, 4.0],
                                        **args):

    p = dataframe.shape[1]
    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])
    dt = dataframe.to_numpy()
    sd_graph = get_prior(graph_prior)
    queue = multiprocessing.Queue()
    processes = []
    rets = []
    for _ in range(reps):
        for T in n_samples:
            for r in randomize:
                for p in pseudo_obs:
                    sd = seqdist.LogLinearJTPosterior()
                    sd.init_model(dt, p, levels, {})
                    print("Starting: " + str((T, r, p, str(sd),
                                              reset_cache, True)))
                    proc = Process(target=trajectory_to_queue,
                                   args=(T, r, sd, sd_graph,
                                         queue, reset_cache, True))
                    processes.append(proc)
                    proc.start()
                    time.sleep(2)

    for _ in processes:
        ret = queue.get() # will block
        rets.append(ret)
    for p in processes:
        p.join()

    output_directory = './'
    output_filename = output_format = None
    if "output_directory" in args:
        output_directory = args["output_directory"]
    if "output_filename" in args:
        output_filename = args["output_filename"]
    if "output_format" in args:
        output_format = args["output_format"]

    for traj in rets:
        aux.write_traj_to_file(traj,
                               dirt=output_directory,
                               output_filename=output_filename,
                               output_format=output_format
                               )
    return rets
        
  

def trajectory_to_queue(n_samples,
                        randomize,
                        seqdist,
                        seqdist_graph,
                        queue,
                        reset_cache=True,
                        reseed=False):

    """ Writes the trajectory of graphs generated by Model to file.
    Args:
        n_samples (int): Number of Gibbs iterations (samples)
        seq_dist (SequentialJTDistributions): the distribution to be sampled from
    Returns:
        Trajectory: Markov chain of underlying graphs of the junction trees sampled by pgibbs.
    """
    if reseed is True:
        np.random.seed()
    graph_trajectory = sample_trajectory(n_samples,
                                         randomize,
                                         seqdist,
                                         seqdist_graph,
                                         reset_cache=reset_cache)
    queue.put(graph_trajectory)




def ggm_loglikelihood(dataframe,
                      graph,
                      D=None,
                      delta=1.0,
                      graph_prior=['mbc', 2.0, 4.0],
                      cache={}, **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta)
    sd_graph = get_prior(graph_prior)
    jt = dlib.junction_tree(graph)
    seps = jtlib.separators(jt)
    clqs = jt.nodes()
    logl = sd.log_likelihood(jt)
    logl += sd_graph.log_prior(clqs, seps)
    return logl
