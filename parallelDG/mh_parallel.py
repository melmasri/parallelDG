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
import parallelDG.graph.parrallel_moves as ndlib
import networkx as nx

# starting MCMC sampler
def sample_trajectory(n_samples,
                      randomize,
                      sd,
                      init_graph=None,
                      reset_cache=True):
    if init_graph:
        graph = init_graph
    else:   
        graph = nx.Graph()
        graph.add_nodes_from(range(sd.p))

    if reset_cache is True:
        sd.cache = {}
    # graph = ar_graph.copy()
    cli_hyper_param = 2
    sep_hyper_param = 1
    jt = dlib.junction_tree(graph)
    assert (jtlib.is_junction_tree(jt))
    jt.latent = True
    jt_traj = [None] * n_samples
    ## graphs = [None] * n_samples
    jt_traj[0] = jt
    ## graphs[0] = jtlib.graph(jt)
    log_prob_traj = [None] * n_samples

    def log_exponentail_markov_law(a, clique):
        return -a * len(clique)
    
    gtraj = mcmctraj.Trajectory()
    gtraj.trajectory_type = "junction_tree"
    gtraj.set_sampling_method({"method": "mh",
                               "params": {"samples": n_samples,
                                          "randomize_interval": randomize}
                               })

    gtraj.set_sequential_distribution(sd)

    log_prob_traj[0] = 0.0
    log_prob_traj[0] = sd.log_likelihood(jtlib.graph(jt_traj[0]))
    # log_prob_traj[0] += -jtlib.log_n_junction_trees(jt_traj[0],
    #                                                 jtlib.separators(jt_traj[0]))
    ## accept_traj = [0] * n_samples
    num_nodes = len(graph)
    k = 0.0
    tic = time.time()
    for i in tqdm(range(1, n_samples), desc="Metropolis-Hastings samples"):
        if i % randomize == 0:
            #jt = dlib.junction_tree(jtlib.graph(jt))
            jtlib.randomize(jt)
            # graphs[i] = jtlib.graph(jt)  # TODO: Improve.
            # log_prob_traj[i] = sd.log_likelihood(graphs[i])
        # A move
        node = frozenset([np.random.randint(num_nodes)])
        move_type = np.random.randint(2)
        
        log_p = 0
        if move_type == 0:          # connect
            #new_cliques, log_q12, N, k = ndlib.propose_connect_moves(jt,node)

            new_moves = ndlib.paritioned_connect_moves(jt, node, True)
            for cl1, cl2_list in new_moves.iteritems():
                for cl2 in cl2_list:
                    cl2_new = cl2 | node
                    if cl2_new in jt:
                        continue
                    sp_new = cl1 & cl2_new
                    sp = cl1 & cl2
                    # New clique log-lik + prior
                    log_p2 = sd.log_likelihood_partial(cliques=[cl2_new],
                                                       separators={sp_new: set([(cl1, cl2_new)])})
                    log_g2 = log_exponentail_markov_law(cli_hyper_param, cl2_new) \
                        - log_exponentail_markov_law(sep_hyper_param, sp_new)
                    # old clique log-like + prior
                    log_p1 = sd.log_likelihood_partial([cl2], {sp: set([(cl1, cl2)])})
                    log_g1 = log_exponentail_markov_law(cli_hyper_param, cl2) \
                        - log_exponentail_markov_law(sep_hyper_param, sp)
                    ## update probability
                    alpha = min(1, np.exp(log_p2 + log_g2 - log_p1 - log_g1))
                    k += 1
                    if np.random.uniform() <= alpha:
                        ndlib.connect(jt, cl2, cl2_new, cl1)
                        log_p += (log_p2 - log_p1)
                        
        else:                       # diconnect
            new_moves = ndlib.paritioned_disconnect_moves(jt, node)
            for cl, conn in new_moves.iteritems():
                if conn:
                    sp = cl & conn
                    cl_new = cl - node
                    if cl_new in jt:
                        continue
                    sp_new = cl_new & conn
                    ## New clique log-lik + prior
                    log_p2 = sd.log_likelihood_partial([cl_new],
                                                       {sp_new:set([(conn, cl_new)])})
                    log_g2 = log_exponentail_markov_law(cli_hyper_param, cl_new) \
                        - log_exponentail_markov_law(sep_hyper_param, sp_new)
                    ## old clique + prior
                    log_p1 = sd.log_likelihood_partial([cl], {sp: set([(conn, cl)])})
                    log_g1 = log_exponentail_markov_law(cli_hyper_param, cl) \
                        - log_exponentail_markov_law(sep_hyper_param, sp)
                    alpha = min(1, np.exp(log_p2 + log_g2 - log_p1 - log_g1))
                    k += 1
                    if np.random.uniform() <= alpha:
                        ndlib.disconnect(jt, cl, cl_new)
                        log_p += (log_p2 - log_p1)

        #accept_traj[i] = 1
        # graphs[i] = jtlib.graph(jt)  # TODO: Improve.
        jt_traj[i] = jt.copy()
        #log_prob_traj[i] = sd.log_likelihood(graphs[i])
        log_prob_traj[i] = log_prob_traj[i-1] + log_p
        # if np.abs(log_prob_traj[i-1] + log_p - log_prob_traj[i]) > 1e-4:
        #     import pdb; pdb.set_trace()
    toc = time.time()
    # gtraj.set_trajectory(graphs)
    gtraj.set_trajectory(jt_traj)
    gtraj.n_updates = k
    gtraj.logl = log_prob_traj
    gtraj.set_time(toc-tic)
    print('Total of {} updates, for an average of {:.2f} per iteration'.format(k, k/n_samples))
    return gtraj


def sample_trajectory_ggm(dataframe, n_samples, randomize=100,
                          D=None, delta=1.0, cache={}, **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, cache)
    return sample_trajectory(n_samples, randomize, sd)

def sample_trajectories_ggm_to_file(dataframe,
                                    n_samples,
                                    randomize=[100],
                                    delta=[1.0],
                                    D=None,
                                    reset_cache=True,
                                    reps=1,
                                    output_directory=".",
                                    **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
   
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
                                                          reset_cache=reset_cache,
                                                          dir=output_directory)
                    graph_trajectories.append(graph_trajectory)
    return graph_trajectories


def sample_trajectories_ggm_parallel(dataframe,
                                     n_samples,
                                     randomize=[100],
                                     D=None,
                                     delta=[1.0,],
                                     reset_cache=True,
                                     reps=1,
                                     **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)

    queue = multiprocessing.Queue()
    processes = []
    rets = []
    for _ in range(reps):
        for T in n_samples:
            for r in randomize:
                for d in delta:
                    sd = seqdist.GGMJTPosterior()
                    sd.init_model(np.asmatrix(dataframe), D, d, {})
                    print("Starting: " + str((T, r, str(sd), reset_cache, True)))
                    proc = Process(target=trajectory_to_queue,
                                   args=(T, r, sd,
                                         queue, reset_cache, True))
                    processes.append(proc)
                    proc.start()
                    time.sleep(2)

    for _ in processes:
        ret = queue.get() # will block
        rets.append(ret)
    for p in processes:
        p.join()

    if "output_directory" in args:
        dir = args["output_directory"]
        for traj in rets:
            write_to_file(traj, dir)
    return rets



def sample_trajectory_loglin(dataframe,
                             n_samples,
                             randomize,
                             pseudo_obs=1.0,
                             reset_cache=True, **args):
    p = dataframe.shape[1]
    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])

    sd = seqdist.LogLinearJTPosterior()
    sd.init_model(dataframe.to_numpy(), pseudo_obs, levels, {})
    return sample_trajectory(n_samples, randomize, sd, reset_cache=reset_cache)


def sample_trajectories_loglin_to_file(dataframe,
                                       n_samples,
                                       randomize=[100, ],
                                       pseudo_obs=[1.0, ],
                                       reset_cache=True,
                                       reps=1,
                                       output_directory=".",
                                       **args):
    p = dataframe.shape[1]
    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])
    dt = dataframe.to_numpy()
   
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
                                                          reset_cache=reset_cache,
                                                          dir=output_directory)
                    graph_trajectories.append(graph_trajectory)
    return graph_trajectories

def sample_trajectories_loglin_parallel(dataframe,
                                        n_samples,
                                        randomize=[100, ],
                                        pseudo_obs=[1.0, ],
                                        reset_cache=True,
                                        reps=1,
                                        **args):

    p = dataframe.shape[1]
    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])
    dt = dataframe.to_numpy()
    
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
                                   args=(T, r, sd,
                                         queue, reset_cache, True))
                    processes.append(proc)
                    proc.start()
                    time.sleep(2)

    for _ in processes:
        ret = queue.get() # will block
        rets.append(ret)
    for p in processes:
        p.join()

    if "output_directory" in args:
        dir = args["output_directory"]
        for traj in rets:
            write_to_file(traj, dir)
    return rets

        
def trajectory_to_file(n_samples,
                       randomize,
                       seqdist,
                       reset_cache=True,
                       dir=".",
                       reseed=False):
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
    graph_trajectory = sample_trajectory(n_samples, randomize,
                                         seqdist, reset_cache=reset_cache)
    write_to_file(graph_trajectory, dir)

    return graph_trajectory


def write_to_file(graph_trajectory, dir):
    date = datetime.datetime.today().strftime('%Y%m%d%H%m%S')
    if not os.path.exists(dir):
        os.mkdir(dir)

    filename = dir + "/" + str(graph_trajectory) + "_" + date + ".json"

    graph_trajectory.write_file(filename=filename)
    print("wrote file: " + filename)
   

def trajectory_to_queue(n_samples,
                        randomize,
                        seqdist,
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
                                         reset_cache=reset_cache)
    queue.put(graph_trajectory)


def max_likelihood_gmm(dataframe, graph, delta=1.0, jt=None):
    p = dataframe.shape[1]
    D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, {})
    # graph = ar_graph.copy()
    if not jt:
        jt = dlib.junction_tree(graph)
        assert (jtlib.is_junction_tree(jt))
    else:
        graph = jtlib.graph(jt)

    loglike_jt = sd.log_likelihood(graph) - \
        jtlib.log_n_junction_trees(jt, jtlib.separators(jt))

    
    return loglike_jt, sd.log_likelihood(graph)



def gen_ggm_trajectory(dataframe, n_samples, D=None, delta=1.0, cache={}, alpha=0.5, beta=0.5, **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, cache)
    return mh(alpha, beta, n_samples, sd)
