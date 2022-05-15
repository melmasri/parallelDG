
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
from parallelDG.graph import graph as glib
import parallelDG.graph.decomposable as dlib
import parallelDG.graph.parallel_moves as pmlib

class Trajectory:
    """
    Class for handling trajectories of decomposable graphical models.
    """
    def __init__(self):
        self.trajectory = []
        self.jt_trajectory = []
        self.time = 0.0
        self.seqdist = None
        self.burnin = 0
        self.logl = []
        self._size = []
        self._jtsize = []
        self.n_updates = 0
        self.jt_updates = []    # format [(move_type(0,1, node, new_clique, old_clique, iteration #))]
        self.graph_updates = []  # format [(edge, edge, movetype(0,1)),] 0 disconnect, 1 connect
        self.init_graph = None
        self.init_jt = None

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

    def set_nupdates(self, n_updates):
        self.n_updates = n_updates

    def set_logl(self, logl):
        self.logl = logl

    def set_jt_updates(self, update_traj):
        self.jt_updates = update_traj

    def set_graph_updates(self, update_traj):
        self.graph_updates = update_traj

    def set_init_graph(self, init_graph):
        self.init_graph = init_graph
        self.init_jt = dlib.junction_tree(init_graph)

    def set_init_jt(self, init_jt):
        self.init_jt = init_jt
        self.init_graph = jtlib.graph(init_jt)
    

    def jt_to_graph_updates(self):
        # (move_type, node, new_clq, old_clq, anchor_clq, iteration)
        g = list()
        # g0 = jtlib.graph(jt.copy())
        # g.append(g0.copy())
        for m in self.jt_updates:
            if m[0] == 0:  # connect move
                g += pmlib.jt_to_graph_connect_move(m[3], m[2], m[1], m[5])
            else:
                g += pmlib.jt_to_graph_disconnect_move(m[3], m[4], m[1], m[5])
        self.set_graph_updates(g)


    def set_graph_trajectories(self):
        if not self.graph_updates:
            self.jt_to_graph_updates()
        g = self.init_graph
        graph_traj = [None] * self.sampling_method['params']['samples']
        graph_traj[0] = g.copy()
        for move in self.graph_updates:
            if move[2] == 0:  # connect
                g.add_edge(*move[:2])
            if move[2] == 1:  # disconnect
                g.remove_edge(*move[:2])
            graph_traj[move[3]] = g.copy()
            # Fill forward graph traj
        for idx, val in enumerate(graph_traj):
            if idx == 0:
                continue  # skip the first element
            if not val:
                graph_traj[idx] = graph_traj[idx-1]
        self.trajectory = graph_traj

    def set_jt_trajecotries(self):
        """ Expiremental"""
        if not self.jt_updates:
            raise TypeError
        self.jt_trajectory = [None] * self.sampling_method['params']['samples']
        self.jt_trajectory[0] = self.init_jt.copy()
        jt = self.init_jt
        trajs = []
        try:
            for move in self.jt_updates:
                if move[3] in jt:
                    if move[0] == 0: # connect
                        pmlib.connect(jt, move[3], move[2], move[4])
                    if move[0] == 1: # disconnect
                        pmlib.disconnect(jt, move[3], move[2])
                trajs.append(jt.copy)
            self.jt_trajectory = trajs
        except:
            import pdb
            pdb.set_trace()

    
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
        if not self.trajectory:
            self.set_graph_trajectories()
            
        for g in self.trajectory[from_index:]:
            graph_dist.add_graph(g, 1./length)
                
        return graph_dist

    def log_likelihood(self, from_index=0):
        if not self.logl:
            if self.trajectory:
                self.logl = [self.seqdist.log_likelihood(g) if g else None for g in self.trajectory]
            elif self.jt_trajectory:
                self.logl = [self.seqdist.log_likelihood(g) if g else None for g in self.jt_trajectory]
            else:
                raise Warning("No trajectory is set")
        return pd.Series(self.logl[from_index:]).fillna(method='ffill')

    def maximum_likelihood_graph(self):
        if not self.trajectory:
            self.set_graph_trajectories()
        ml_ind = self.log_likelihood().idxmax()
        return self.trajectory[ml_ind]


    def jtsize(self, from_index=0):
        """ Plots the auto-correlation function of the graph size (number of edges)
        Args:
            from_index (int): Burn-in period, default=0.
        """
        if not self._jtsize and self.jt_trajectory:
            self._jtsize = [g.size() if g else None for g in self.jt_trajectory[from_index:]]
        return pd.Series(self._jtsize).fillna(method='ffill')
        
    def size(self, from_index=0):
        """ Plots the auto-correlation function of the graph size (number of edges)
        Args:
            from_index (int): Burn-in period, default=0.
        """
        if not self._size:
            if not self.trajectory:
                self.set_graph_trajectories()
            self._size = [g.size() if g else None for g in self.trajectory[from_index:]]
        return pd.Series(self._size).fillna(method='ffill')

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

    def graph_diff_trajectory_df(self):

        def list_to_string(edge_list):
            s = "["
            for i, e in enumerate(edge_list):
                s += str(e[0]) + "-" + str(e[1])
                if i != len(edge_list)-1:
                    s += ";"
            return s + "]"

        added = []
        removed = []

        for i in range(1, self.init_graph.order()):
            added += [(0, i)]
        
        df0 = pd.DataFrame({"index": [-2],
                            "added": [list_to_string(added)],
                            "removed": [list_to_string([])],
                            "score": [0]})
        df1 = pd.DataFrame({"index": [-1],
                            "added": [list_to_string([])],
                            "removed": [list_to_string(added)],
                            "score": [0]})
        df2 = df0.append(df1)

        if not self.graph_updates:
            self.jt_to_graph_updates()

        output = pd.DataFrame(self.graph_updates,
                              columns=['e1', 'e2', 'move_type', 'index'])
        output['edge_tupple'] = output[["e1","e2"]].apply(tuple, axis=1)
        formated_output = output.groupby(['move_type',
                                          'index'])['edge_tupple'].apply(
                                              list_to_string).reset_index(
                                                  name='edges')

        added = formated_output[output['move_type'] == 0].rename(
            columns={"edges": "added",
                     "index": "index"})

        removed = formated_output[output['move_type'] == 1].rename(
            columns={"edges": "removed",
                     "index": "index"})

        removed['added'] = None
        added['removed'] = None

        _cols = ['added', 'index', 'removed']
        df = added[_cols].append(removed[_cols]).sort_values(
            by='index'
        ).fillna('[]')
        #  getting loglikelihood
        score = pd.DataFrame({
            'index': range(len(self.logl)),
            'score': self.logl
        })

        final_df = df.merge(score)

        return df2.append(final_df)

    
    def to_json(self, optional={}):
        mcmc_traj = {"model": self.seqdist.get_json_model(),
                     "run_time": self.time,
                     "optional": optional,
                     "sampling_method": self.sampling_method,
                     "n_updates": self.n_updates,
                     "graph_updates": self.graph_updates,
                     "jt_updates": self.jt_updates,
                     "init_graph": json_graph.node_link_data(self.init_graph)
                     }
        return mcmc_traj


    def from_json(self, mcmc_json):
        self.set_time(mcmc_json["run_time"])
        self.sampling_method = mcmc_json["sampling_method"]
        self.n_updates = mcmc_json['n_updates']
        self.optional = mcmc_json["optional"]
        self.init_graph = jtlib.to_frozenset(
            json_graph.node_link_graph(
                mcmc_json['init_graph']))
        self.seqdist.init_model_from_json(mcmc_json["model"])
        self.n_updates = mcmc_json['n_updates']
        self.set_graph_updates(mcmc_json['graph_updates'])
        self.set_jt_updates(mcmc_json['jt_updates'])
        
        if mcmc_json["model"]["name"] == "ggm_jt_post":
            self.seqdist = sd.GGMJTPosterior()
        elif mcmc_json["model"]["name"] == "loglin_jt_post":
            self.seqdist = sd.LogLinearJTPosterior()


    def read_file(self, filename):
        """ Reads a trajectory from json-file.
        """
        with open(filename) as mcmc_file:
            mcmc_json = json.load(mcmc_file)
        self.from_json(mcmc_json)

    def __str__(self):
        self.sampling_method["method"] == "mh":
        return "mh_graph_trajectory_" + str(self.seqdist) + "_length_" +        str(len(self.trajectory)) + \
            "_randomize_interval_" + str(self.sampling_method["params"]["randomize_interval"])

