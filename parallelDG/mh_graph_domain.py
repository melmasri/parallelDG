from multiprocessing import Process
import multiprocessing
import datetime
import time
import os

import numpy as np
import networkx as nx
from tqdm import tqdm

import parallelDG.distributions.sequential_junction_tree_distributions as seqdist
import parallelDG.graph.trajectory as mcmctraj
import parallelDG.graph.junction_tree as jtlib
import parallelDG.graph.decomposable as dlib
import parallelDG.graph.graph_domain as aglib
from parallelDG import auxiliary_functions as aux
import parallelDG.graph.parallel_moves as ndlib

def sample_trajectory_single_move(n_samples, randomize, sd, **args):
    graph = nx.Graph()
    seed = args.get('seed', time.time())
    np.random.seed(seed)
    graph.add_nodes_from(range(sd.p))
    jt = dlib.junction_tree(graph)
    assert (jtlib.is_junction_tree(jt))
    jt_traj = [None] * n_samples
    graphs = [None] * n_samples
    jt_traj[0] = jt
    graphs[0] = jtlib.graph(jt)
    log_prob_traj = [None] * n_samples
    num_nodes = sd.p
    jt.latent = False
    r0 = [None] * n_samples
    r1 = [None] * n_samples
    
    gtraj = mcmctraj.Trajectory()
    gtraj.set_sampling_method({"method": "mh",
                               "params": {"samples": n_samples,
                                          "randomize_interval": randomize}
                               })

    gtraj.set_sequential_distribution(sd)
    log_prob_traj[0] = 0.0
    log_prob_traj[0] = sd.log_likelihood(jt_traj[0])
    log_prob_traj[0] += -jtlib.log_n_junction_trees(jt_traj[0], jtlib.separators(jt_traj[0]))

    accept_traj = [0] * n_samples

    MAP_graph = (graphs[0], log_prob_traj[0])

    for i in tqdm(range(1, n_samples), desc="Metropolis-Hastings samples"):
        if log_prob_traj[i-1] > MAP_graph[1]:
            MAP_graph = (graphs[i-1], log_prob_traj[i-1])

        if i % randomize == 0:
            jtlib.randomize(jt)
            graphs[i] = jtlib.graph(jt)  # TODO: Improve.
            log_prob_traj[i] = sd.log_likelihood(jt) - jtlib.log_n_junction_trees(jt, jtlib.separators(jt))

        r = np.random.randint(2)  # Connect / disconnect move
        num_seps = jt.size()
        num_cliques = jt.order()
        log_p1 = log_prob_traj[i - 1]
        node = frozenset([np.random.randint(num_nodes)])
        if r == 0:
            # Connect move
            partitions, num_partitions = ndlib.paritioned_connect_moves(jt, node)
            np.random.shuffle(partitions)
            #print('----')
            #print(jt.edges())
            if not partitions:
                log_prob_traj[i] = log_prob_traj[i-1]
                graphs[i] = graphs[i-1]
                continue
            sep = partitions[0]
            #  edgeind = np.random.randint(jt.size())
            #  sep = list(jt.edges())[edgeind]
            #  node = frozenset([np.random.choice(list(sep[0] - sep[1]))])
            num_cliques = jt.order()
            conn = aglib.connect_move(jt, sep, node)  # need to move to calculate posterior
            seps_prop = jtlib.separators(jt)
            log_p2 = sd.log_likelihood(jt) - jtlib.log_n_junction_trees(jt, seps_prop)
            _, num_revers_partitions = ndlib.paritioned_disconnect_moves(jt, node)  # + len(ndlib.paritioned_connect_moves(jt, node)) 
            log_p2 -= np.log(num_partitions)
            log_p1 -= np.log(num_revers_partitions)
            C_disconn = conn[2] | conn[3] | conn[4]
            if conn[0] == "a":
                (case, log_q12, X, Y, S, CX_disconn, CY_disconn, XSneig, YSneig) = conn
                (NX_disconn, NY_disconn, N_disconn) = aglib.disconnect_get_neighbors(jt, C_disconn, X, Y)  # TODO: could this be done faster?
                #import pdb; pdb.set_trace()
                if  NX_disconn: 
                    sep_disconn = (list(NX_disconn)[0], C_disconn)
                else:
                    sep_disconn = (frozenset([]), C_disconn)
                log_q21 = aglib.disconnect_logprob_a(num_cliques - 1, X, Y, S, N_disconn, sep_disconn)
                #print log_p2, log_q21, log_p1, log_q12
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                #print alpha
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    # print "Accept"
                    accept_traj[i] = 1
                    log_prob_traj[i] = log_p2 + np.log(num_partitions)
                    graphs[i] = jtlib.graph(jt)  # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.disconnect_a(jt, C_disconn, X, Y, CX_disconn, CY_disconn, XSneig, YSneig)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

            elif conn[0] == "b":
                (case, log_q12, X, Y, S, CX_disconn, CY_disconn) = conn
                sep_disconn = (CX_disconn, Y | X | S)
                log_q21 = aglib.disconnect_logprob_bcd(num_cliques, X, Y, S, sep_disconn)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])

                if samp == 1:
                    #print "Accept"
                    accept_traj[i] = 1
                    log_prob_traj[i] = log_p2 + np.log(num_partitions)
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.disconnect_b(jt, C_disconn, X, Y, CX_disconn, CY_disconn)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

            elif conn[0] == "c":
                (case, log_q12, X, Y, S, CX_disconn, CY_disconn) = conn
                (NX_disconn, NY_disconn, N_disconn) = aglib.disconnect_get_neighbors(jt, C_disconn, X, Y)  # TODO: could this be done faster?
                if  NX_disconn: 
                    sep_disconn = (list(NX_disconn)[0], C_disconn)
                else:
                    sep_disconn = (frozenset([]), X | Y | S)
                log_q21 = aglib.disconnect_logprob_bcd(num_cliques, X, Y, S, sep_disconn)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    accept_traj[i] = 1
                    #print "Accept"
                    log_prob_traj[i] = log_p2 + np.log(num_partitions)
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.disconnect_c(jt, C_disconn, X, Y, CX_disconn, CY_disconn)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

            elif conn[0] == "d":
                (case, log_q12, X, Y, S,  CX_disconn, CY_disconn) = conn
                sep_disconn = (CX_disconn, X | Y | S)
                log_q21 = aglib.disconnect_logprob_bcd(num_cliques + 1, X, Y, S, sep_disconn)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    accept_traj[i] = 1
                    #print "Accept"
                    log_prob_traj[i] = log_p2 + np.log(num_partitions)
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.disconnect_d(jt, C_disconn, X, Y, CX_disconn, CY_disconn)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

        elif r == 1:
            # Disconnect move
            partitions, num_partitions = ndlib.paritioned_disconnect_moves(jt, node)
            np.random.shuffle(partitions)
            if not partitions:
                log_prob_traj[i] = log_prob_traj[i-1]
                graphs[i] = graphs[i-1]
                #import pdb;pdb.set_trace()
                continue
            sep = partitions[0]
            #  GT selection
            # C = np.random.choice(list(jt.nodes()))
            # if len(C) < 2:
            #     log_prob_traj[i] = log_prob_traj[i-1]
            #     graphs[i] = graphs[i-1]
            #     continue
            # jtn = list(jt.neighbors(C))
            # if jtn:
            #     C1 = np.random.choice(jtn)
            #     sep = (C1, C)
            #     if C1 & C:
            #         node = frozenset([np.random.choice(list(sep[0] & sep[1]))])
            #     else:
            #         node = frozenset([np.random.choice(list(C))])
            # else:
            #     sep = (frozenset([]), C)
            #     node = frozenset([np.random.choice(list(C))])
            
            #  import pdb; pdb.set_trace()
            disconnect = aglib.disconnect_move(jt, sep, node)
            #import pdb; pdb.set_trace() 
            seps_prop = jtlib.separators(jt)
            log_p2 = sd.log_likelihood(jt) - jtlib.log_n_junction_trees(jt, seps_prop)
            _, num_revers_partitions = ndlib.paritioned_connect_moves(jt, node)
            log_p2 -= np.log(num_partitions)
            log_p1 -= np.log(num_revers_partitions)
            if not disconnect: 
                log_prob_traj[i] = log_prob_traj[i-1]
                graphs[i] = graphs[i-1]
                continue

            if disconnect[0] == "a":
                (case, log_q12, X, Y, S, CX_conn, CY_conn) = disconnect
                log_q21 = aglib.connect_logprob(num_seps + 1, X, Y, CX_conn, CY_conn)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    accept_traj[i] = 1
                    #print "Accept"
                    log_prob_traj[i] = log_p2 + np.log(num_partitions)
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.connect_a(jt, S, X, Y, CX_conn, CY_conn)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

            elif disconnect[0] == "b":
                (case, log_q12, X, Y, S, CX_conn, CY_conn) = disconnect
                log_q21 = aglib.connect_logprob(num_seps, X, Y, CX_conn, CY_conn)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    accept_traj[i] = 1
                    #print "Accept"
                    log_prob_traj[i] = log_p2 + np.log(num_partitions)
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.connect_b(jt, S, X, Y, CX_conn, CY_conn)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

            elif disconnect[0] == "c":
                (case, log_q12, X, Y, S, CX_conn, CY_conn) = disconnect
                log_q21 = aglib.connect_logprob(num_seps, X, Y, CX_conn, CY_conn)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    accept_traj[i] = 1
                    #print "Accept"
                    log_prob_traj[i] = log_p2 + np.log(num_partitions)
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.connect_c(jt, S, X, Y, CX_conn, CY_conn)
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue

            elif disconnect[0] == "d":
                (case, log_q12, X, Y, S, CX_conn, CY_conn) = disconnect
                log_q21 = aglib.connect_logprob(num_seps - 1, X, Y, CX_conn, CY_conn)
                alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
                samp = np.random.choice(2, 1, p=[(1 - alpha), alpha])
                if samp == 1:
                    #print "Accept"
                    accept_traj[i] = 1
                    log_prob_traj[i] = log_p2 + np.log(num_partitions)
                    graphs[i] = jtlib.graph(jt) # TODO: Improve.
                else:
                    #print "Reject"
                    aglib.connect_d(jt, S, X, Y, CX_conn, CY_conn)
                
                    log_prob_traj[i] = log_prob_traj[i-1]
                    graphs[i] = graphs[i-1]
                    continue
        # print("r {}, #moves {}, #r-moves {}, #ratio {:2f}".format(r,
        #                                                        num_partitions,
        #                                                        num_revers_partitions,
#                                                               float(num_partitions)/num_revers_partitions))
        #    print(np.mean(accept_traj[:i]))
        if r==0:
            r0[i] = float(num_partitions)/num_revers_partitions
        if r==1:
            r1[i] = float(num_partitions)/num_revers_partitions
            
    gtraj.set_trajectory(graphs)
    gtraj.logl = log_prob_traj
    return gtraj


def sample_trajectory_uniform(n_samples, randomize=100, graph_size=5, cache={}, **args):
    sd = seqdist.UniformJTDistribution(graph_size)
    return sample_trajectory_single_move(n_samples=n_samples,
                             randomize=randomize,
                             sd=sd)


def sample_trajectory_loglin(dataframe, n_samples, pseudo_obs=1.0, randomize=1000, cache={}, **args):

    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])
    sd = seqdist.LogLinearJTPosterior()
    sd.init_model(dataframe.get_values(), pseudo_obs, levels, cache_complete_set_prob=cache)

    return sample_trajectory_single_move(n_samples, randomize, sd)


def sample_trajectories_loglin_to_file(dataframe, n_samples, randomize=[1000], pseudo_obs=[1.0],
                                    reps=1, output_directory=".", **args):

    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])

    graph_trajectories = []
    for _ in range(reps):
        for r in randomize:
            for T in n_samples:
                sd = seqdist.LogLinearJTPosterior()
                sd.init_model(dataframe.get_values(), pseudo_obs, levels)

                graph_trajectory = trajectory_to_file(T, r, sd, dir=output_directory)
                graph_trajectories.append(graph_trajectory)
    return graph_trajectories


def sample_trajectories_loglin_parallel(dataframe, n_samples, randomize=[1000], pseudo_obs=[1.0],
                                     reps=1, output_directory=".", **args):

    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])
    queue = multiprocessing.Queue()
    processes = []
    rets = []

    for _ in range(reps):
        for r in randomize:
            for T in n_samples:
                sd = seqdist.LogLinearJTPosterior()
                sd.init_model(dataframe.get_values(), pseudo_obs, levels)
                print("Starting: " + str((T, r, str(sd), True)))

                proc = Process(target=trajectory_to_queue,
                               args=(T, r,
                                     sd, queue, True))
                proc.start()
                processes.append(proc)
                time.sleep(2)

    for _ in processes:
        ret = queue.get()  # will block
        rets.append(ret)
    for p in processes:
        p.join()

    return rets


def trajectory_to_file(n_samples, randomize, seqdist, dir=".", reseed=False):
    """ Writes the trajectory of graphs generated by particle Gibbs to file.

    Args:
        seq_dist (SequentialJTDistributions): the distribution to be sampled from
        filename_prefix (string): prefix to the filename

    Returns:
        mcmctraj.Trajectory: Markov chain of underlying graphs of the junction trees.

    """
    if reseed is True:
        np.random.seed()

    print(n_samples, str(randomize), str(seqdist))
    graph_trajectory = sample_trajectory_single_move(n_samples, randomize, seqdist)
    date = datetime.datetime.today().strftime('%Y%m%d%H%m%S')
    if not os.path.exists(dir):
        os.mkdir(dir)

    filename = dir + "/" + str(graph_trajectory) +"_"+ date + ".json"
    graph_trajectory.write_file(filename=filename)
    print("wrote file: " + filename)

    return graph_trajectory

def trajectory_to_queue(n_samples, randomize, seqdist, queue, reseed=False):
    """ Writes the trajectory of graphs generated by particle Gibbs to file.

    Args:
        seq_dist (SequentialJTDistributions): the distribution to be sampled from
        filename_prefix (string): prefix to the filename

    Returns:
        mcmctraj.Trajectory: Markov chain of underlying graphs of the junction trees

    """
    if reseed is True:
        np.random.seed()

    print(n_samples, str(randomize), str(seqdist))
    graph_trajectory = sample_trajectory_single_move(n_samples, randomize, seqdist)
    queue.put(graph_trajectory)

def sample_trajectory_ggm(dataframe, n_samples, randomize=1000, D=None, delta=1.0, cache={}, **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, cache)
    return sample_trajectory_single_move(n_samples, randomize, sd, **args)


def sample_trajectories_ggm_to_file(dataframe, n_samples, randomize=[1000], D=None, delta=1.0,
                                    reps=1, output_directory=".", **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)

    graph_trajectories = []
    for _ in range(reps):
        for r in randomize:
            for T in n_samples:
                sd = seqdist.GGMJTPosterior()
                sd.init_model(np.asmatrix(dataframe), D, delta, {})
                graph_trajectory = trajectory_to_file(T, r, sd, dir=output_directory)
                graph_trajectories.append(graph_trajectory)
    return graph_trajectories


def sample_trajectories_ggm_parallel(dataframe,
                                     n_samples,
                                     randomize=[1000],
                                     D=None,
                                     delta=1.0,
                                     reps=1,
                                     **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    queue = multiprocessing.Queue()
    processes = []
    rets = []

    for _ in range(reps):
        for r in randomize:
            for T in n_samples:
                sd = seqdist.GGMJTPosterior()
                sd.init_model(np.asmatrix(dataframe), D, delta, {})

                print("Starting: " + str((T, r, str(sd), True)))

                proc = Process(target=trajectory_to_queue,
                               args=(T, r,
                                     sd, queue, True))
                proc.start()
                processes.append(proc)
                time.sleep(2)

    for _ in processes:
        ret = queue.get() # will block
        rets.append(ret)
    for p in processes:
        p.join()

    if "output_directory" in args:
        dir = args["output_directory"]
        for traj in rets:
            aux.write_traj_to_file(traj, dir)
    return rets
