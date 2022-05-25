
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
        self.graph_dist = None

    def set_sampling_method(self, method):
        self.sampling_method = method

    def set_sequential_distribution(self, seqdist):
        """ Set the SequentialJTDistribution for the graphs in the trajectory

        Args:
            seqdist (SequentialJTDistribution): A sequential distribution
        """
        self.seqdist = seqdist

    def set_graph_prior(self, graph_dist):
        """ Set the graph prior distribution prior

        Args: 
            graph_dist (SequentialJTDistribution): A sequential distribution
        """
        self.graph_dist = graph_dist

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
        # (iteration, move_type, node, (new_clq, old_clq, anchor_clq))
        g = list()
        # g0 = jtlib.graph(jt.copy())
        # g.append(g0.copy())
        for m in self.jt_updates:
            if m[1] == 0:  # connect move
                g += pmlib.jt_to_graph_connect_move(m[3], m[2], m[0])
            else:               # disconnect
                g += pmlib.jt_to_graph_disconnect_move(m[3], m[2], m[0])
        self.set_graph_updates(g)

    def set_graph_trajectories(self):
        if not self.graph_updates:
            self.jt_to_graph_updates()
        g = self.init_graph
        graph_traj = [None] * self.sampling_method['params']['samples']
        graph_traj[0] = g.copy()
        for move in self.graph_updates:
            if move[1] == 0:  # connect
                g.add_edge(*move[2])
            if move[1] == 1:  # disconnect
                g.remove_edge(*move[2])
            graph_traj[move[0]] = g.copy()
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

        for move in self.jt_updates:
            if move[3] in jt:
                if move[0] == 0:  # connect
                    pmlib.connect(jt, move[3], move[2], move[4])
                if move[0] == 1:  # disconnect
                    pmlib.disconnect(jt, move[3], move[2])
            trajs.append(jt.copy)
        self.jt_trajectory = trajs

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
                self.logl = [self.seqdist.log_likelihood(dlib.junction_tree(g)) if g else None for g in self.trajectory]
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
                e = sorted(e)
                s += str(e[0]) + "-" + str(e[1])
                if i != len(edge_list)-1:
                    s += ";"
            return s + "]"

        def flatten(t):
            return [item for sublist in t for item in sublist]

        def list_to_string_flatten(t):
            return list_to_string(flatten(t))

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

        # Creating raw + summary updates
        output = pd.DataFrame(self.graph_updates,
                              columns=['index',
                                       'move_type',
                                       'edge_tuple'])
        prev_edges_set = edges_set = set([])
        added_list = list()
        removed_list = list()
        added_raw_list = list()
        removed_raw_list = list()
        for index, row in output.iterrows():
            prev_edges_set = edges_set
            if row['move_type'] == 0:  # connection
                edges_set = edges_set | set([tuple(sorted(row['edge_tuple']))])
                added_raw_list.append((row['index'], [row['edge_tuple']]))
            if row['move_type'] == 1:  # disconnect
                edges_set = edges_set - set([tuple(sorted(row['edge_tuple']))])
                removed_raw_list.append((row['index'], [row['edge_tuple']]))
            added =  list(edges_set - prev_edges_set)
            removed = list(prev_edges_set - edges_set)
            if added:
                added_list.append((row['index'], added))
            if removed:
                removed_list.append((row['index'], removed))
        # Raw list
        # added_raw = pd.DataFrame(added_raw_list,
        #                          columns=['index',
        #                                   'added']
        #                          ).groupby(
        #                             ['index']
        #                          ).added.apply(
        #                              list_to_string_flatten
        #                          ).reset_index(name='added')
        # added_raw['removed'] = None
        # removed_raw = pd.DataFrame(removed_raw_list,
        #                            columns=['index',
        #                                     'removed']
        #                            ).groupby(
        #                                ['index']
        #                            ).removed.apply(
        #                                list_to_string_flatten
        #                            ).reset_index(name='removed')
        # removed_raw['added'] = None
        # _cols = ['added', 'index', 'removed']
        # df_raw = added_raw[_cols].append(removed_raw[_cols]).sort_values(
        #     by='index').fillna('[]')
        # df_raw.rename(columns={'index': 'index_raw',
        #                        'added': 'added_raw',
        #                        'removed': 'removed_raw'
        #                        },
        #               inplace=True)
        # Summary list
        added = pd.DataFrame(added_list,
                             columns=['index',
                                      'added']
                             ).groupby(
                                 ['index']
                             ).added.apply(
                                 list_to_string_flatten
                             ).reset_index(name='added')
        added['removed'] = None
        removed = pd.DataFrame(removed_list,
                               columns=['index',
                                        'removed']
                               ).groupby(
                                   ['index']
                               ).removed.apply(
                                   list_to_string_flatten
                               ).reset_index(name='removed')
        removed['added'] = None
        _cols = ['added', 'index', 'removed']
        df = added[_cols].append(removed[_cols]).sort_values(
            by='index').fillna('[]')

        #  getting loglikelihood
        score = pd.DataFrame({
            'index': range(len(self.logl)),
            'score': self.logl
        })

        final_df = df.merge(score, how='left')
        # final_raw_df = df_raw.merge(score,
        #                             how='left',
        #                             left_on='index_raw'
        #                             right_on='index'
        #                            )
        # df2.rename(columns={
        #     'index': 'index_raw',
        #     'added': 'added_raw',
        #     'removed': 'removed_raw',
        #     'score': 'score_raw'
        #   }).append(final_raw_df)
        return df2.append(final_df)

    def to_json(self, optional={}):
        mcmc_traj = {"model": self.seqdist.get_json_model(),
                     "run_time": self.time,
                     "optional": optional,
                     "sampling_method": self.sampling_method,
                     "n_updates": self.n_updates,
                     "graph_updates": self.graph_updates,
                     "jt_updates": self.jt_updates,
                     "init_graph": json_graph.node_link_data(self.init_graph),
                     "loglikelihood_trace": self.logl
        }
        return mcmc_traj

  
    def from_json(self, mcmc_json):

        def jt_update_to_frozenset(update):
            return (update[0],  # mcmc iteration
                    update[1],  # move type 0 connect 1 disconnect
                    frozenset(update[2]),  # node
                    (frozenset(update[3][0]),  # new clique
                     frozenset(update[3][1]),  # old clique
                     frozenset(update[3][2])  # anchor clique
                     )
                    )   
        self.set_time(mcmc_json["run_time"])
        self.sampling_method = mcmc_json["sampling_method"]
        self.n_updates = mcmc_json['n_updates']
        self.optional = mcmc_json["optional"]
        self.init_graph = json_graph.node_link_graph(
            mcmc_json['init_graph'])
        self.logl = mcmc_json['loglikelihood_trace']
        self.n_updates = mcmc_json['n_updates']
        self.set_graph_updates(mcmc_json['graph_updates'])
        self.set_jt_updates(map(jt_update_to_frozenset,
                                mcmc_json['jt_updates']))
        
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
        strl = (
            "mh_graph_trajectory_" + str(self.seqdist)
            + "_length_" + str(len(self.trajectory))
            + "_randomize_interval_"
            + str(self.sampling_method["params"]["randomize_interval"])
        )
        return strl

