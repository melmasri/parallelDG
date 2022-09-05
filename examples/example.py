"""
Examples for graph inference for continuous data and discrete data.
The data (and all the belonging parameters) are either be simulated
or read from an external file.
"""
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


import parallelDG.auxiliary_functions as aux
import parallelDG.graph.decomposable as dlib
import parallelDG.distributions.g_intra_class as gic
import parallelDG.mh_greenthomas as green
import parallelDG.distributions.discrete_dec_log_linear as loglin
from parallelDG.graph import trajectory as mcmctraj
import parallelDG.mh_parallel as pdg


# Discrete data
# reads labels and support from rows 0 and 1 respectively
np.random.seed(2)
aw_df = pd.read_csv("sample_data/czech_autoworkers.csv", header=[0, 1])
graph_trajs = pdg.sample_trajectories_loglin_parallel(dataframe=aw_df,
                                                      n_samples=[10000],
                                                      randomize=[100],
                                                      reset_cache=False,
                                                      reps=2)

aux.plot_multiple_traj_statistics(graph_trajs,
                                  0,
                                  write_to_file=True,
                                  output_directory="./aw_trajs/",
                                  annot=True)



## Continuous AR(1-5)-model
np.random.seed(2)
ar_graph = dlib.sample_random_AR_graph(50, 5)
cov_mat = gic.cov_matrix(ar_graph, 0.9, 1.0)
ar_df = pd.DataFrame(np.random.multivariate_normal(np.zeros(50), cov_mat, 100))

# Parallel MH algorithm
graph_trajs = pdg.sample_trajectories_ggm_parallel(dataframe=ar_df,
                                                   n_samples=[10000],
                                                   randomize=[100],
                                                   reset_cache=True,
                                                   reps=2)

aux.plot_multiple_traj_statistics(graph_trajs,
                                  burnin_end=5000,
                                  write_to_file=True,
                                  output_directory="./ar_1-5_trajs/")



# 15 nodes log-linear data
loglin_graph = nx.Graph()
loglin_graph.add_nodes_from(range(15))
loglin_graph.add_edges_from([(0, 11), (0, 7), (1, 8), (1, 6), (2, 4), (3, 8), (3, 9),
                       (3, 10), (3, 4), (3, 6), (4, 6), (4, 8), (4, 9), (4, 10),
                       (5, 10), (5, 6), (6, 8), (6, 9), (6, 10), (7, 11), (8, 9),
                       (8, 10), (8, 11), (9, 10), (10, 11), (12, 13)])
np.random.seed(1)
levels = np.array([range(2)] * loglin_graph.order())
loglin_table = loglin.sample_prob_table(loglin_graph, levels, 1.0)
np.random.seed(5)
loglin_df = pd.DataFrame(loglin.sample(loglin_table, 1000))
loglin_df.columns = [list(range(loglin_graph.order())), [len(l) for l in levels]]

graph_trajs = pdg.sample_trajectories_loglin_parallel(dataframe=loglin_df,
                                                      n_samples=[10000],
                                                      randomize=[100],
                                                      reps=2)

aux.plot_multiple_traj_statistics(graph_trajs,
                                  0,
                                  write_to_file=True,
                                  output_directory="./loglin_trajs/")


# Metropolis-Hastings algorithm from
# P. J. Green and A. Thomas. Sampling decomposable graphs using a Markov chain on junction trees. Biometrika, 100(1):91-110, 2013.
graph_trajs = green.sample_trajectories_ggm_parallel(dataframe=ar_df, randomize=[100],
                                                     n_samples=[10000],
                                                     reset_cache=True,
                                                     reps=2)

aux.plot_multiple_traj_statistics(graph_trajs,
                                  0,
                                  write_to_file=True,
                                  output_directory="./ar_1-5_trajs_green/")

