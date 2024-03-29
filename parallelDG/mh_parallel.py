# """
# Metropolis-Hastings Parallel clique sampler
# """
from multiprocessing import Process
import multiprocessing

import time
import datetime
import os
import numpy as np
from tqdm import tqdm
import math
import parallelDG.distributions.sequential_junction_tree_distributions as seqdist
import parallelDG.graph.graph as glib
import parallelDG.graph.junction_tree as jtlib
import parallelDG.graph.trajectory as mcmctraj
import parallelDG.graph.decomposable as dlib
import parallelDG.graph.parallel_moves as ndlib
import networkx as nx
import parallelDG.auxiliary_functions as aux

# starting MCMC sampler
def sample_trajectory_single_move(n_samples,
                                  randomize,
                                  sd,
                                  sd_graph,
                                  init_graph=None,
                                  reset_cache=True,
                                  **args):
    seed = args.get('seed', int(time.time()))
    reduced_jt = args.get('reduced_jt', False)
    np.random.seed(seed)
    if init_graph:
        graph = init_graph.copy()
        jt = dlib.junction_tree(graph)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(range(sd.p))
        dummy_tree = jtlib.to_frozenset(nx.random_tree(n=sd.p, seed=seed))
        jt = jtlib.JunctionTree()
        jt.add_nodes_from(dummy_tree.nodes())
        jt.add_edges_from(dummy_tree.edges())
        jt.num_graph_nodes = sd.p
    sd.cache = {}
    
    t = jtlib.JunctionMap(jt)
    log_prob_traj = [0] * n_samples
    p = t.p
    gtraj = mcmctraj.Trajectory()
    gtraj.trajectory_type = "Latent Juction tree"
    gtraj.set_sampling_method({"method": "Metropolis-Hastings - single move sampler",
                               "params": {"samples": n_samples,
                                          "randomize_interval": randomize}
                               })
    gtraj.set_sequential_distribution(sd)
    gtraj.set_graph_prior(sd_graph)
    gtraj.set_init_graph(graph) 
    gtraj.set_init_jt(jt)
    log_prob_traj[0] = sd.log_likelihood_partial(t.get_cliques(), t.get_separators())
    log_prob_traj[0] += sd_graph.log_prior(t.get_cliques(), t.get_separators(), 0)
    update_moves = list()
    t_moves = list()
    k = 1
    tic = time.time()
    acc_ratios = []
    for i in tqdm(range(1, n_samples), desc="Metropolis-Hastings - single-move"):
        if i % randomize == 0:
            #pass
            #t.randomize()
            #t.randomize_at_sep_single()
            #t.randomize_bfs()
            t.randomize_by_jt() 
        node = np.random.randint(p)
        move_type = np.random.randint(2)
        log_p = 0.0
        ratios= 0.0
        subtree_set = t.node2t[node]
        subtree_n_nodes = int(len(subtree_set))
        ## assigning move type to the only possble moves.
        if subtree_n_nodes == 0:
            move_type = 0
        if subtree_n_nodes == t.t.order():
            move_type = 1
            
        if move_type == 0:
            moves, num_moves = ndlib.propose_connect_moves_set(t.t, subtree_set)
            np.random.shuffle(moves)
            if not moves:
                print('not move connect')
                log_prob_traj[i] = log_prob_traj[i-1]
                continue
            reverse_moves, num_reverse_moves1 = ndlib.propose_disconnect_moves_set(t.t, subtree_set, True)
            num_reverse_moves = ndlib.reverse_move_leafs_count(reverse_moves, moves[0][1])
            num_reverse_moves += 1 * (subtree_n_nodes == 1)
            U, Uadj = moves[0]
            C =  t.t2clique[U]
            if isinstance(Uadj, int): 
                Cadj = t.t2clique[Uadj]
            else:
                Cadj = []
            Cadj = frozenset(Cadj)
            Cnew = frozenset(C | {node})
            C = frozenset(C)
            S =  frozenset(C & Cadj)
            Snew = frozenset(Cnew & Cadj)

            log_p2 = sd.log_likelihood_partial([Cnew],{Snew: [(Cadj, Cnew)]})
            log_p1 = sd.log_likelihood_partial([C], {S: [(Cadj, C)]})
            log_g2 = sd_graph.log_prior_partial(Cnew, Snew, move_type)
            log_g1 = sd_graph.log_prior_partial(C, S, 1-move_type)
            log_q2 = -np.log(num_moves) 
            log_q1 = -np.log(num_reverse_moves)
            
            if subtree_n_nodes == 0:
                log_p1 += sd.log_likelihood_partial([frozenset([node])], {S: [(Cadj, C)]})
                log_g1 += sd_graph.log_prior_partial(frozenset([node]), S, 1-move_type)
                
            ratios = (log_p2 - log_p1, log_g2 - log_g1, log_q2 - log_q1)
            #acc_ratios.append(ratios)
            k += int(1)
            if np.random.uniform() <= min(1, np.exp(np.sum(ratios))):
                log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                update_moves.append((i, k, move_type, {node}, (Cnew, C, Cadj)))
                #t_moves.append((i, k, move_type, node, U))
                t.connect(U, node)
        else:
            moves, num_moves = ndlib.propose_disconnect_moves_set(t.t, subtree_set)
            np.random.shuffle(moves)
            if not moves:
                print('not move disconnect')
                log_prob_traj[i] = log_prob_traj[i-1]
                continue
            if subtree_n_nodes == 1:
                num_reverse_moves = t.t.order()  # 1e-10 to not move
            else:
                reverse_moves, num_reverse_moves1 = ndlib.propose_connect_moves_set(t.t, subtree_set)
                num_reverse_moves = ndlib.reverse_neighbors_count(t.t, moves[0][0], num_reverse_moves1)
            U, Uadj = moves[0]
            C = t.t2clique[U]
            if isinstance(Uadj, int): 
                Cadj = t.t2clique[Uadj]
            else:
                Cadj = []
            Cadj = frozenset(Cadj)
            Cnew = frozenset(C - {node})
            C = frozenset(C)
            S =  frozenset(C & Cadj)
            Snew = frozenset(Cnew & Cadj)

            log_p2 = sd.log_likelihood_partial([Cnew],{Snew: [(Cadj, Cnew)]})
            log_p1 = sd.log_likelihood_partial([C], {S: [(Cadj, C)]})
            log_g2 = sd_graph.log_prior_partial(Cnew, Snew, move_type)
            log_g1 = sd_graph.log_prior_partial(C, S, 1-move_type)
            log_q2 = -np.log(num_moves)
            log_q1 = -np.log(num_reverse_moves)
            if subtree_n_nodes == 1:
                log_p2 += sd.log_likelihood_partial([frozenset([node])],{Snew: [(Cadj, Cnew)]})
                log_g2 += sd_graph.log_prior_partial(frozenset([node]), Snew, 1)
                
            ratios = (log_p2 - log_p1, log_g2 - log_g1, log_q2 - log_q1) 
            #acc_ratios.append(ratios)
            k += int(1)
            if np.random.uniform() <= min(1, np.exp(np.sum(ratios))):
                log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                update_moves.append((i, k, move_type, {node}, (Cnew, C, Cadj)))
                #t_moves.append((i, k, move_type, node, U))
                t.disconnect(U, node)
        log_prob_traj[i] = log_prob_traj[i-1] + log_p
       
    toc = time.time()
    gtraj.set_logl(log_prob_traj)
    gtraj.set_nupdates(k)
    gtraj.set_time(toc-tic)
    gtraj.set_jt_updates(update_moves)
    gtraj.dummy = acc_ratios
    gtraj.t = t
    #gtraj.t_updates = t_moves
    print('Total of {} updates, for an average of {:.2f} per iteration or {:.2f}updates/sec'.format(k, float(k)/n_samples,k/(toc-tic)))
    print('Acceptance rate {:.4f}'.format(float(len(update_moves))/k))
    return gtraj
                
def sample_trajectory(n_samples,
                      randomize,
                      sd,
                      sd_graph,
                      init_graph=None,
                      **args):
    seed = args.get('seed', int(time.time()))
    np.random.seed(seed)
    if init_graph:
        graph = init_graph.copy()
        jt = dlib.junction_tree(graph)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(range(sd.p))
        dummy_tree = jtlib.to_frozenset(nx.random_tree(n=sd.p, seed=seed))
        jt = jtlib.JunctionTree()
        jt.add_nodes_from(dummy_tree.nodes())
        jt.add_edges_from(dummy_tree.edges())
        jt.num_graph_nodes = sd.p

    sd.cache = {}
    t = jtlib.JunctionMap(jt)
    log_prob_traj = [0] * n_samples
    p = int(t.p)
    gtraj = mcmctraj.Trajectory()
    gtraj.trajectory_type = "Latent Juction tree"
    gtraj.set_sampling_method({"method": "Metropolis-Hastings - parallel sampler",
                               "params": {"samples": n_samples,
                                          "randomize_interval": randomize}
                               })
    gtraj.set_sequential_distribution(sd)
    gtraj.set_graph_prior(sd_graph)
    gtraj.set_init_graph(graph) 
    gtraj.set_init_jt(jt)
    
    log_prob_traj[0] = sd.log_likelihood_partial(t.get_cliques(), t.get_separators())
    log_prob_traj[0] += sd_graph.log_prior(t.get_cliques(), t.get_separators(), 1)
    update_moves = list()
    num_nodes = sd.p
    k = int(0)
    tic = time.time()
    acc_ratios = []

    for i in tqdm(range(1, n_samples), desc="Metropolis-Hastings - parallel moves"):
        if i % randomize == 0:
            #t.randomize()
            t.randomize_by_jt()
            #t.randomize_at_sep_single()
        node = np.random.randint(p)
        move_type = np.random.randint(2)
        log_p = 0.0
        ratios= 0.0
        subtree_set = t.node2t[node]
        subtree_n_nodes = int(len(subtree_set))
        ## assigning move type to the only possble moves.
        if subtree_n_nodes == 0:
            move_type = 0
        if subtree_n_nodes == p:
            move_type = 1
        if move_type == 0:
            moves, num_moves = ndlib.propose_connect_moves_set(t.t, subtree_set)
            if not moves:
                log_prob_traj[i] = log_prob_traj[i-1]
                continue
            for U, Uadj in moves:
                C =  t.t2clique[U]
                if isinstance(Uadj, int): 
                    Cadj = t.t2clique[Uadj]
                else:
                    Cadj = []
                C = frozenset(C)
                Cadj = frozenset(Cadj)
                Cnew = frozenset(C | {node})
                S =  frozenset(C & Cadj)
                Snew = frozenset(Cnew & Cadj)
                log_p2 = sd.log_likelihood_partial([Cnew],{Snew: [(Cadj, Cnew)]})
                log_p1 = sd.log_likelihood_partial([C], {S: [(Cadj, C)]}) 
                log_g2 = sd_graph.log_prior_partial(Cnew, Snew, 0)
                log_g1 = sd_graph.log_prior_partial(C, S, 1)
                if subtree_n_nodes == 0:
                    log_p1 += sd.log_likelihood_partial([frozenset([node])], {S: [(Cadj, C)]})
                    log_g1 += sd_graph.log_prior_partial(frozenset([node]), S, 1)
                    
                ratios = (log_p2 - log_p1, log_g2 - log_g1, 0)
                #acc_ratios.append((i,) + ratios)
                k += int(1)
                if np.random.uniform() <= min(1, np.exp(np.sum(ratios))):
                    log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                    update_moves.append((i, k, move_type, {node}, (Cnew, C, Cadj)))
                    t.connect(U, node)
        else:
            moves, num_moves = ndlib.propose_disconnect_moves_set(t.t, subtree_set)
            if not moves:
                log_prob_traj[i] = log_prob_traj[i-1]
                continue
            log_q = -np.log(2) if subtree_n_nodes == 2 else 0.0
            for U, Uadj in moves:
                C =  t.t2clique[U]
                if isinstance(Uadj, int): 
                    Cadj = t.t2clique[Uadj]
                else:
                    Cadj = []
                C = frozenset(C)
                Cadj = frozenset(Cadj)
                Cnew = frozenset(C - {node})
                S =  frozenset(C & Cadj)
                Snew = frozenset(Cnew & Cadj)
                log_p2 = sd.log_likelihood_partial([Cnew],{Snew: [(Cadj, Cnew)]})
                log_p1 = sd.log_likelihood_partial([C], {S: [(Cadj, C)]})
                log_g2 = sd_graph.log_prior_partial(Cnew, Snew, 1)
                log_g1 = sd_graph.log_prior_partial(C, S, 0)
                if subtree_n_nodes == 1:
                    log_p2 += sd.log_likelihood_partial([frozenset([node])],{Snew: [(Cadj, Cnew)]})
                    log_g2 += sd_graph.log_prior_partial(frozenset([node]), Snew, 1)
                
                ratios = (log_p2 - log_p1, log_g2 - log_g1, log_q)
                #acc_ratios.append((i, ) + ratios)
                k += int(1)
                if np.random.uniform() <= min(1, np.exp(np.sum(ratios))):    
                    log_p += (log_p2 - log_p1) + (log_g2 - log_g1)
                    update_moves.append((i, k, move_type, {node}, (Cnew, C, Cadj)))
                    t.disconnect(U, node)
        log_prob_traj[i] = log_prob_traj[i-1] + log_p
       
    toc = time.time()
    gtraj.set_logl(log_prob_traj)
    gtraj.set_nupdates(k)
    gtraj.set_time(toc-tic)
    gtraj.set_jt_updates(update_moves)
    gtraj.dummy = acc_ratios
    gtraj.t = t
    print('Total of {} updates, for an average of {:.2f} per iteration or {:.2f}updates/sec'.format(k, float(k)/n_samples,k/(toc-tic)))
    print('Acceptance rate {:.4f}'.format(float(len(update_moves))/k))
    return gtraj

def get_prior(graph_prior):
    graph_prior_type = graph_prior[0].lower()
    default_parameters = {
        "mbc": (seqdist.ModifiedBornnCaron, [2.0, 4.0]),
        "edgepenalty": (seqdist.EdgePenalty, [0.001]),
        "jumppenalty": (seqdist.JumpPenalty, [0.25]),
        "uniform": (seqdist.GraphUniform, [])
    }
    if graph_prior_type not in default_parameters:
        graph_prior_type = "mbc"
        
    sd_class, default_vals = default_parameters[graph_prior_type]
    # If more parameters are provided in graph_prior, use them instead of defaults
    parameters = graph_prior[1:] if len(graph_prior) > 1 else default_vals
    parameters = parameters[:len(default_vals)]
    return sd_class(*map(float, parameters))


def sample_trajectory_uniform(
        n_samples,
        randomize=100,
        graph_prior=['uniform'],
        graph_size=10,
        cache={}, **args):
    sd = seqdist.UniformJTDistribution(graph_size)
    sd_graph = get_prior(graph_prior)

    single_move = args.get('single_move', False)

    if not single_move:
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
                          **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, cache = {})
    sd_graph = get_prior(graph_prior)

    parallel = args.get('parallel', False)
    if not parallel:
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
                       output_directory=".",
                       reseed=False,
                       labels=None,
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

    parallel = args.get('parallel', False)
    #print (n_particles, alpha, beta, radius, n_samples, str(seqdist), reset_cache)
    if parallel:
        graph_trajectory = sample_trajectory(n_samples,
                                             randomize,
                                             seqdist,
                                             seqdist_graph,
                                             reset_cache=True,
                                             **args)

    else:
        graph_trajectory = sample_trajectory_single_move(n_samples,
                                                         randomize,
                                                         seqdist,
                                                         seqdist_graph,
                                                         reset_cache=True,
                                                         **args)
    output_filename = args.get("output_filename", None)
    seed = args.get('seed', int(time.time()))
    if not output_filename:
        output_filename = 'graph_traj_'+str(seed)+'.csv'
    output_format = args.get("output_format", None)
    aux.write_traj_to_file(graph_trajectory=graph_trajectory,
                           dirt=output_directory,
                           output_filename=output_filename,
                           output_format=output_format,
                           labels=labels
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
    node_labels = np.array(dataframe.columns.get_level_values(0))
    
    for _ in range(reps):
        for T in n_samples:
            for r in randomize:
                for d in delta:
                    sd = seqdist.GGMJTPosterior()
                    sd.init_model(np.asmatrix(dataframe), D, d, cache={})
                    graph_trajectory = trajectory_to_file(n_samples=T,
                                                          randomize=r,
                                                          seqdist=sd,
                                                          seqdist_graph=sd_graph,
                                                          output_directory=output_directory,
                                                          labels=node_labels,
                                                          **args)
                    graph_trajectories.append(graph_trajectory)
    return graph_trajectories


def sample_trajectories_ggm_parallel(dataframe,
                                     n_samples,
                                     randomize=[100],
                                     D=None,
                                     delta=[1.0,],
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
    seed = args.get('seed', time.time())
    for _ in range(reps):
        for T in n_samples:
            for r in randomize:
                for d in delta:
                    sd = seqdist.GGMJTPosterior()
                    sd.init_model(np.asmatrix(dataframe), D, d, cache={})
                    print("Starting: " + str((T, r, str(sd),  True)))
                    proc = Process(target=trajectory_to_queue,
                                   args=(T, r, sd, sd_graph,
                                         queue,  True))
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
    sd.init_model(dataframe.to_numpy(), pseudo_obs, levels, {}, {})
    sd_graph = get_prior(graph_prior)
    if 'single_move' in args:
        return sample_trajectory_single_move(n_samples,
                                             randomize,
                                             sd,
                                             sd_graph,
                                             reset_cache=reset_cache,
                                             **args)
    else:
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
    node_labels = np.array(dataframe.columns.get_level_values(0))
    
    for _ in range(reps):
        for T in n_samples:
            for r in randomize:
                for p in pseudo_obs:
                    sd = seqdist.LogLinearJTPosterior()
                    sd.init_model(dt, p, levels, {}, {})   
                    graph_trajectory = trajectory_to_file(n_samples=T,
                                                          randomize=r,
                                                          seqdist=sd,
                                                          seqdist_graph=sd_graph,
                                                          reset_cache=reset_cache,
                                                          output_directory=output_directory,
                                                          labels=node_labels,
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
                    sd.init_model(dt, p, levels, {}, {})
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
                                         seqdist_graph)
                                       
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
    sd.init_model(np.asmatrix(dataframe), D, delta, cache)
    sd_graph = get_prior(graph_prior)
    jt = dlib.junction_tree(graph)
    seps = jtlib.separators(jt)
    clqs = jt.nodes()
    logl = sd.log_likelihood(jt)
    logl += sd_graph.log_prior(clqs, seps)
    return logl


def loglinear_loglekihood(dataframe,
                          graph,
                          pseudo_obs=[1.0, ],
                          graph_prior=['mbc', 2.0, 4.0],
                          **args):
    p = dataframe.shape[1]
    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])
    dt = dataframe.to_numpy()

    sd = seqdist.LogLinearJTPosterior()
    sd.init_model(dataframe.to_numpy(),
                  pseudo_obs,
                  levels, {}, {})
    sd_graph = get_prior(graph_prior)
    
    jt = dlib.junction_tree(graph)
    seps = jtlib.separators(jt)
    clqs = jt.nodes()
    logl = sd.log_likelihood(jt)
    logl += sd_graph.log_prior(clqs, seps)
    return logl
