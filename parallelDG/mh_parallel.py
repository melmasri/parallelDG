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
import parallelDG.graph.parallel_moves as ndlib
import networkx as nx
import parallelDG.auxiliary_functions as aux

import copy
# starting MCMC sampler
def sample_trajectory_single_move(n_samples,
                                  randomize,
                                  sd,
                                  sd_graph,
                                  init_graph=None,
                                  reset_cache=True,
                                  **args):
    seed = args.get('seed', time.time())
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

    jt.clique_hard_threshold = args.get('jt.clique_hard_threshold', sd.p)
    if reset_cache is True:
        sd.cache = {}

    jt.latent = True
    jt_traj = [None] * n_samples
    jt_traj[0] = jt.copy()
    log_prob_traj = [0] * n_samples

    gtraj = mcmctraj.Trajectory()
    gtraj.trajectory_type = "junction_tree"
    gtraj.set_sampling_method({"method": "mh",
                               "params": {"samples": n_samples,
                                          "randomize_interval": randomize}
                               })
    gtraj.set_sequential_distribution(sd)
    gtraj.set_graph_prior(sd_graph)
    gtraj.set_init_graph(graph) 
    gtraj.set_init_jt(jt)

    log_prob_traj[0] = sd.log_likelihood(jt_traj[0])
    update_moves = list()
    num_nodes = len(graph)
    k = int(0)
    tic = time.time()
    def _continue(i):
        log_prob_traj[i] = log_prob_traj[i-1]

    def copy_jt(jt):
        jt1 = jt.copy()
        jt1.latent = jt.latent
        jt1.num_graph_nodes = jt.num_graph_nodes
        jt1.clique_hard_threshold = jt.clique_hard_threshold
        return jt1

    # Sampling moves and nodes
    acc_ratios = []
    for i in tqdm(range(1, n_samples), desc="Metropolis-Hastings samples"):
        if i % randomize == 0:
            jtlib.randomize(jt)
        # selecting moves and nodes
        node = frozenset([np.random.randint(num_nodes)])
        move_type = np.random.randint(2)
        log_p = 0.0
        if move_type == 0:          # connect
            partitions, num_partitions = ndlib.paritioned_connect_moves(jt, node)
            np.random.shuffle(partitions)             #  shuffling the output
            if not partitions:
                _continue(i)
                continue
            cl_anchor, cl_current = partitions[0]
            cl_prop = cl_current | node
            if cl_prop in jt:
                _continue(i)
                continue
            sp_prop = cl_anchor & cl_prop
            sp = cl_anchor & cl_current
            # New clique log-lik + prior
            log_p2 = sd.log_likelihood_partial(
                cliques=[cl_prop],
                separators={sp_prop: set([(cl_anchor, cl_prop)])})
            log_p1 = sd.log_likelihood_partial(
                [cl_current], {sp: set([(cl_anchor, cl_current)])})
            log_g2 = sd_graph.log_prior_partial(cl_prop, sp_prop)
            # old clique log-like + prior
            log_g1 = sd_graph.log_prior_partial(cl_current, sp)
            prev_jt = copy_jt(jt)
            ndlib.connect(jt, cl_current, cl_prop, cl_anchor)
            _, num_revers_partitions = ndlib.paritioned_disconnect_moves(jt, node) 
            log_q2 = -np.log(num_partitions)
            log_q1 = -np.log(num_revers_partitions)
            # update probability
            acc_ratios.append((log_p2 - log_p1, log_g2 - log_g1, log_q2 - log_q1))
            alpha = min(1,
                        np.exp(log_p2 + log_g2 + log_q2 - log_p1 - log_g1 - log_q1))
            k += int(1)
            if np.random.uniform() <= alpha:
                # ndlib.connect(jt, cl_current, cl_prop, cl_anchor)
                log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                update_moves.append((i, k, move_type, node, (cl_prop,
                                                             cl_current,
                                                             cl_anchor)))
            else:
                #ndlib.disconnect(jt, cl_prop, cl_current)
                jt = prev_jt

        else:                       # diconnect
            partitions, num_partitions = ndlib.paritioned_disconnect_moves(jt, node)
            np.random.shuffle(partitions)             #  shuffling the output
            if not partitions:
                _continue(i)
                continue
            cl_anchor, cl_current = partitions[0]
            if cl_anchor:
                sp = cl_current & cl_anchor
                cl_prop = cl_current - node
                if cl_prop in jt:
                    _continue(i)
                    continue
                sp_prop = cl_prop & cl_anchor
                # New clique log-lik + prior
                log_p2 = sd.log_likelihood_partial(
                    [cl_prop],
                    {sp_prop: set([(cl_anchor, cl_prop)])})
                log_p1 = sd.log_likelihood_partial([cl_current],
                                                   {sp: set([(cl_anchor, cl_current)])})
                log_g1 = sd_graph.log_prior_partial(cl_current, sp)
                log_g2 = sd_graph.log_prior_partial(cl_prop, sp_prop)
                # old clique + prior
                prev_jt = copy_jt(jt)
                ndlib.disconnect(jt, cl_current, cl_prop)
                _, num_revers_partitions = ndlib.paritioned_connect_moves(jt, node)
                log_q2 = -np.log(num_partitions)
                log_q1 = -np.log(num_revers_partitions)
                acc_ratios.append((log_p2 - log_p1, log_g2 - log_g1, log_q2 - log_q1))
                alpha = min(1,
                            np.exp(log_p2 + log_g2 + log_q2 - log_p1 - log_g1 - log_q1))
                k += int(1)
                if np.random.uniform() <= alpha:
                    #ndlib.disconnect(jt, cl_current, cl_prop)
                    log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                    update_moves.append((i, k, move_type, node, (cl_prop,
                                                                 cl_current,
                                                                 cl_anchor)))
                else:
                    jt = prev_jt
                    # if cl_prop:
                    #     ndlib.connect(jt, cl_prop, cl_current, cl_anchor)
                    # else:
                    #    ndlib.connect(jt, frozenset([]), cl_current, cl_anchor)
            else:                   # disconnect to an empty clique
                sp = sp_prop = sp_single = cl_anchor
                cl_prop = cl_current - node
                cl_single = node
                if cl_prop in jt or cl_single in jt:
                    _continue(i)
                    continue
                # New clique log-lik + prior
                log_p2 = sd.log_likelihood_partial(
                    [cl_prop],
                    {sp_prop: set([(cl_anchor, cl_prop)])})
                log_p2 += sd.log_likelihood_partial(
                    [cl_single],     
                    {sp_single: set([(cl_anchor, cl_single)])})    
                log_p1 = sd.log_likelihood_partial([cl_current],
                                                   {sp: set([(cl_anchor, cl_current)])})
                log_g1 = sd_graph.log_prior_partial(cl_current, sp)
                log_g2 = sd_graph.log_prior_partial(cl_prop, sp_prop)
                log_g2 += sd_graph.log_prior_partial(cl_single, sp_single)
                acc_ratios.append((log_p2 - log_p1, log_g2 - log_g1, 0.0))
                alpha = min(1, np.exp(log_p2 + log_g2 - log_p1 - log_g1))
                k += int(1)
                if np.random.uniform() <= alpha:
                    ndlib.disconnect(jt, cl_current, cl_prop)
                    ndlib.connect(jt, cl_anchor, cl_single, cl_prop)
                    log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                    update_moves.append((i, k,
                                         move_type, node,
                                         (cl_prop,
                                          cl_current,
                                          cl_anchor)))

        log_prob_traj[i] = log_prob_traj[i-1] + log_p
        # jt_traj[i] = jt.copy()
    toc = time.time()
    gtraj.set_logl(log_prob_traj)
    gtraj.set_nupdates(k)
    gtraj.set_time(toc-tic)
    gtraj.set_jt_updates(update_moves)
    # gtraj.jt_trajectory = jt_traj
    # gtraj.trajectory = graphs
    gtraj.dummy = acc_ratios
    print('Total of {} updates, for an average of {:.2f} per iteration or {:.2f}updates/sec'.format(k, float(k)/n_samples,k/(toc-tic)))
    print('Acceptance rate {:.4f}'.format(float(len(update_moves))/k))
    return gtraj


def sample_trajectory(n_samples,
                      randomize,
                      sd,
                      sd_graph,
                      init_graph=None,
                      reset_cache=True,
                      **args):
    seed = args.get('seed', time.time())
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


    jt.clique_hard_threshold = args.get('jt.clique_hard_threshold', sd.p)
    if reset_cache is True:
        sd.cache = {}

    jt.latent = True
    jt_traj = [None] * n_samples
    jt_traj[0] = jt.copy()
    log_prob_traj = [None] * n_samples

    gtraj = mcmctraj.Trajectory()
    gtraj.trajectory_type = "junction_tree"
    gtraj.set_sampling_method({"method": "mh",
                               "params": {"samples": n_samples,
                                          "randomize_interval": randomize}
                               })
    gtraj.set_sequential_distribution(sd)
    gtraj.set_graph_prior(sd_graph)
    gtraj.set_init_graph(graph) 
    gtraj.set_init_jt(jt)

    log_prob_traj[0] = sd.log_likelihood(jt_traj[0])
    update_moves = list()
    num_nodes = len(graph)
    k = int(0)
    acc_ratios = []
    tic = time.time()
    def _continue(i):
        log_prob_traj[i] = log_prob_traj[i-1]
    for i in tqdm(range(1, n_samples), desc="Parallel Metropolis-Hastings samples"):
        if i % randomize == 0:
            jtlib.randomize(jt)

        node = frozenset([np.random.randint(num_nodes)])
        move_type = np.random.randint(2)
        log_p = 0.0
        if move_type == 0:          # connect
            partitions, num_partitions = ndlib.paritioned_connect_moves(jt, node)
            np.random.shuffle(partitions)             #  shuffling the output
            if not partitions:
                _continue(i)
                continue
            for cl_anchor, cl_current in partitions:
                cl_prop = cl_current | node
                if cl_prop in jt:
                    continue
                sp_prop = cl_anchor & cl_prop
                sp = cl_anchor & cl_current
                # New clique log-lik + prior
                log_p2 = sd.log_likelihood_partial(
                    cliques=[cl_prop],
                    separators={sp_prop: set([(cl_anchor, cl_prop)])})
                log_g2 = sd_graph.log_prior_partial(cl_prop, sp_prop)
                # old clique log-like + prior
                log_p1 = sd.log_likelihood_partial(
                    [cl_current], {sp: set([(cl_anchor, cl_current)])})
                log_g1 = sd_graph.log_prior_partial(cl_current, sp)
                acc_ratios.append((i, log_p2 - log_p1, log_g2 - log_g1, 0))
                alpha = min(1, np.exp(log_p2 + log_g2 - log_p1 - log_g1))
                k += int(1)
                if np.random.uniform() <= alpha:
                    ndlib.connect(jt, cl_current, cl_prop, cl_anchor)
                    # graphs.append(jtlib.graph(jt))
                    log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                    update_moves.append((i, k,
                                         move_type,
                                         node,
                                         (cl_prop,
                                          cl_current,
                                          cl_anchor)))
        else:                       # diconnect
            partitions, num_partitions = ndlib.paritioned_disconnect_moves(jt, node)
            np.random.shuffle(partitions)
            if not partitions:
                _continue(i)
                continue
            for cl_anchor, cl_current in partitions:
                if cl_anchor: 
                    sp = cl_current & cl_anchor
                    cl_prop = cl_current - node
                    if cl_prop in jt:
                        continue
                    sp_prop = cl_prop & cl_anchor
                    # New clique log-lik + prior
                    log_p2 = sd.log_likelihood_partial(
                        [cl_prop],
                        {sp_prop: set([(cl_anchor, cl_prop)])})
                    log_g2 = sd_graph.log_prior_partial(cl_prop, sp_prop)
                    # old clique + prior
                    log_p1 = sd.log_likelihood_partial([cl_current],
                                                       {sp: set([(cl_anchor, cl_current)])})
                    log_g1 = sd_graph.log_prior_partial(cl_current, sp)
                    acc_ratios.append((i, log_p2 - log_p1, log_g2 - log_g1, 0))
                    alpha = min(1, np.exp(log_p2 + log_g2 - log_p1 - log_g1))
                    k += int(1)
                    if np.random.uniform() <= alpha:
                        ndlib.disconnect(jt, cl_current, cl_prop)
                        #   graphs.append(jtlib.graph(jt))
                        log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                        update_moves.append((i, k,
                                             move_type,
                                             node,
                                             (cl_prop,
                                              cl_current,
                                              cl_anchor)))
                else:           # disconnect to an empty clique
                    sp = sp_prop = sp_single = cl_anchor
                    cl_prop = cl_current - node
                    cl_single = node
                    if cl_prop in jt or cl_single in jt:
                        continue
                    # New clique log-lik + prior
                    log_p2 = sd.log_likelihood_partial(
                        [cl_prop],
                        {sp_prop: set([(cl_anchor, cl_prop)])})
                    log_p2 += sd.log_likelihood_partial(
                        [cl_single],     
                        {sp_single: set([(cl_anchor, cl_single)])})    
                    
                    log_g2 = sd_graph.log_prior_partial(cl_prop, sp_prop)
                    log_g2 += sd_graph.log_prior_partial(cl_single, sp_single)
                    # old clique + prior
                    log_p1 = sd.log_likelihood_partial([cl_current],
                                                       {sp: set([(cl_anchor, cl_current)])})
                    log_g1 = sd_graph.log_prior_partial(cl_current, sp)
                    acc_ratios.append((i, log_p2 - log_p1, log_g2 - log_g1, 0))
                    alpha = min(1, np.exp(log_p2 + log_g2 - log_p1 - log_g1))
                    k += int(1)
                    if np.random.uniform() <= alpha:
                        ndlib.disconnect(jt, cl_current, cl_prop)
                        ndlib.connect(jt, cl_anchor, cl_single, cl_prop)
                        # graphs.append(jtlib.graph(jt))
                        log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                        update_moves.append((i, k,
                                             move_type, node,
                                             (cl_prop,
                                              cl_current,
                                              cl_anchor)))
        # print("move type {}, q(t,t') {}, q(t', t) {}, alpha {}".format(move_type,
        #                                                                n_moves,
        #                                                                revers_n_moves,
        #                                                                alpha
        # ))
        #import pdb;pdb.set_trace()
        log_prob_traj[i] = log_prob_traj[i-1] + log_p
        # jt_traj[i] = jt.copy()

    toc = time.time()
    gtraj.set_logl(log_prob_traj)
    gtraj.set_nupdates(k)
    gtraj.set_time(toc-tic)
    gtraj.set_jt_updates(update_moves)
    # gtraj.jt_trajectory = jt_traj
    # gtraj.trajectory = graphs
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
    if not output_filename:
        output_filename = 'graph_traj.csv'
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
