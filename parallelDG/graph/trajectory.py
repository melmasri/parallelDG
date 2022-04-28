
"""
A class for handling Markov chains produced from e.g. MCMC.
"""
import json
import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
import numpy as np

import parallelDG.graph.empirical_graph_distribution as gdist
from parallelDG.distributions import sequential_junction_tree_distributions as sd
import parallelDG.graph.junction_tree as jtlib

class Trajectory:
    """
    Class for handling trajectories of decomposable graphical models.
    """
    def __init__(self):
        self.trajectory = []
        self.time = 0.0
        self.seqdist = None
        self.burnin = 0
        self.logl = []
        self._size = []
        self._jtsize = []
        self.n_updates = 0
        self.trajectory_type = 'junction_tree'

    def set_sampling_method(self, method):
        self.sampling_method = method

    def set_sequential_distribution(self, seqdist):
        """ Set the SequentialJTDistribution for the graphs in the trajectory

        Args:
            seqdist (SequentialJTDistribution): A sequential distribution
        """
        self.seqdist = seqdist

    def set_trajectory(self, trajectory):
        """ Set the trajectory of graphs.

        Args:
            trajectory (Trajectory): An MCMC trajectory of graphs.
        """
        self.trajectory = trajectory

    def set_time(self, generation_time):
        self.time = generation_time

    def add_sample(self, graph, time, logl=None):
        """ Add graph to the trajectory.

        Args:
            graph (NetworkX graph):
            time (list): List of times it took to generate each sample
        """
        self.trajectory.append(graph)
        self.time.append(time)
        if logl is not None:
            self.logl.append(logl)

    def empirical_distribution(self, from_index=0):
        length = len(self.trajectory) - from_index
        graph_dist = gdist.GraphDistribution()
        if nx.is_tree(self.trajectory[0]):
            for g in self.trajectory[from_index:]:
                graph_dist.add_graph(jtlib.graph(g), 1./length)
        else:
            for g in self.trajectory[from_index:]:
                graph_dist.add_graph(g, 1./length)
        return graph_dist

    def log_likelihood(self, from_index=0):
        if self.logl == []:
            if self.trajectory_type=="junction_tree":
                self.logl = [self.seqdist.log_likelihood(jtlib.graph(g)) for g in self.trajectory]
            else:
                self.logl = [self.seqdist.log_likelihood(g) for g in self.trajectory]
        return pd.Series(self.logl[from_index:])

    def maximum_likelihood_graph(self):
        ml_ind = self.log_likelihood().idxmax()
        if nx.is_tree(self.trajectory[ml_ind]):
            return jtlib.graph(self.trajectory[ml_ind])
        else:
            return self.trajectory[ml_ind]


    def jtsize(self, from_index=0):
        """ Plots the auto-correlation function of the graph size (number of edges)
        Args:
            from_index (int): Burn-in period, default=0.
        """
        if self._jtsize ==[]:
            self._jtsize = [g.size() for g in self.trajectory[from_index:]]
        return pd.Series(self._jtsize)
        
    def size(self, from_index=0):
        """ Plots the auto-correlation function of the graph size (number of edges)
        Args:
            from_index (int): Burn-in period, default=0.
        """
        if self._size == []:
            if nx.is_tree(self.trajectory[0]):
                return self.jtsize(from_index)
            else:
                self._size = [jtlib.graph(g).size() for g in self.trajectory[from_index:]]
        return pd.Series(self._size)

    def write_file(self, filename=None, optional={}):
        """ Writes a Trajectory together with the corresponding
        sequential distribution to a json-file.
        """
        
        def default(o):
            if isinstance(o, np.int64) or isinstance(o, np.float):
                return o
            if isinstance(o, frozenset):
                return list(o)
            raise TypeError

        if filename is None:
            with open(str(self) + ".json", 'w') as outfile:
                json.dump(self.to_json(optional=optional), outfile, default=default)
        else:
            with open(filename, 'w') as outfile:
                json.dump(self.to_json(optional=optional), outfile, default=default)


    def to_json(self, optional={}):
        js_graphs = [json_graph.node_link_data(graph) for
                     graph in self.trajectory]

        mcmc_traj = {"model": self.seqdist.get_json_model(),
                     "run_time": self.time,
                     "optional": optional,
                     "sampling_method": self.sampling_method,
                     "trajectory": js_graphs,
                     "n_updates": self.n_updates,
                     "trajectory_type": self.trajectory_type
                     }
        return mcmc_traj


    def from_json(self, mcmc_json):
        graphs = [
            jtlib.to_frozenset(json_graph.node_link_graph(js_graph))
            for js_graph in mcmc_json["trajectory"]]

        self.set_trajectory(graphs)
        self.set_time(mcmc_json["run_time"])
        self.sampling_method = mcmc_json["sampling_method"]
        self.n_updates = mcmc_json['n_updates']
        self.optional = mcmc_json["optional"]
        self.trajectory_type = mcmc_json["trajectory_type"]

        if mcmc_json["model"]["name"] == "ggm_jt_post":
            self.seqdist = sd.GGMJTPosterior()
        elif mcmc_json["model"]["name"] == "loglin_jt_post":
            self.seqdist = sd.LogLinearJTPosterior()

        self.seqdist.init_model_from_json(mcmc_json["model"])

    def read_file(self, filename):
        """ Reads a trajectory from json-file.
        """
        with open(filename) as mcmc_file:
            mcmc_json = json.load(mcmc_file)

        self.from_json(mcmc_json)

    def __str__(self):
        if self.sampling_method["method"] == "pgibbs":
            return "pgibbs_graph_trajectory_" + str(self.seqdist) + "_length_" + str(len(self.trajectory)) + \
            "_N_" + str(self.sampling_method["params"]["N"])
        elif self.sampling_method["method"] == "mh":
            return "mh_graph_trajectory_" + str(self.seqdist) + "_length_" + str(len(self.trajectory)) + \
                "_randomize_interval_" + str(self.sampling_method["params"]["randomize_interval"])


