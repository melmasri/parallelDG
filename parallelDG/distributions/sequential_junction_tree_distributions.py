"""
Junction tree distributions suitable for sampling.
"""

import numpy as np
import networkx as nx
import parallelDG.graph.decomposable
from parallelDG.distributions import gaussian_graphical_model
from parallelDG.distributions import discrete_dec_log_linear as loglin
import parallelDG.graph.junction_tree as jtlib


class SequentialJTDistribution(object):
    """
    Abstract class of junction tree distributions for SMC sampling.
    """

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        pass

    def __str__(self):
        pass


class ModifiedBornnCaron(SequentialJTDistribution):
    """ Graph prior ratios based on modified Bornn & Caron graph priors.
        As P(G) = prod{clq} f(clq)/ prod{sep} g(sep)
        f(x) = exp(alpha * |x| -1)
        g(x) = exp(beta * |x|)
    """

    def __init__(self, param_clq=4.0, param_sep=2.0):
        self.param_clq = param_clq
        self.param_sep = param_sep

    def log_potential(self, param, clq_size):
        return param * (clq_size)
    
    def log_prior(self, cliques, seperators):
        lclq = 0.0
        lsep = 0.0
        for c in cliques:
            lclq += self.log_potential(self.param_clq, len(c) - 1.0)
        for s in seperators:
            lsep += self.log_potential(self.param_sep, len(s))
        return lclq - lsep

    def log_prior_partial(self, clq, sep):
        lp = 0.0
        lp += self.log_potential(self.param_clq, len(clq) - 1.0)
        lp -= self.log_potential(self.param_sep, len(sep))
        return lp

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        new = self.log_prior(new_cliques, new_separators)
        old = self.log_prior(old_cliques, old_separators)
        return new - old


class JumpPenalty(SequentialJTDistribution):
    """  Penalizes the move type as exp(-alpha) for connect and exp(alpha) for connect
    """
    def __init__(self, alpha=0.25):
        self.alpha = alpha
    
    def log_prior_partial(self, clq, sep):
        return -self.alpha 

    def log_prior(self, clqs, seps):
        return -self.alpha * len(clqs)
        
    
    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        return 0.0


class RandomWalkPenalty(SequentialJTDistribution):
    """  Penalizes the the random walk
    """
    def __init__(self, alpha= 0.5):
        self.alpha = alpha

    def log_prior(self, cliques, seperators):
        return 0.0

    def log_prior_partial(self, clq, sep, **args):
        if 'disconnect' in args:
            return np.log(1-self.alpha)
        return np.log(self.alpha)
        
    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        return 0.0



class GraphUniform(SequentialJTDistribution):
    """  Uniform sampling over decomposable graphs, i.e pi(G) ~ 1/|jt(G)|
    """
    def __init__(self):
        None

    def log_prior(self, cliques, seperators):
        return 0.0

    def log_prior_partial(self, clq, sep):
        return 0.0

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        return 0.0


class EdgePenalty(SequentialJTDistribution):
    """ Edge penalty prior, as P(G) = prod{clq, sep} f(clq) / f(sep). 
        f(x) = exp(- alpha * |x|(|x| -1))
    """

    def __init__(self, alpha=0.001):
        self.alpha = alpha

    def log_potential(self, c):
        a = len(c)
        return - self.alpha * a * (a - 1.0)/2.0

    def log_prior(self, cliques, seperators):
        lclq = 0.0
        lsep = 0.0
        for c in cliques:
            lclq += self.log_potential(c)
        for s in seperators:
            lsep += self.log_potential(s)
        return lclq - lsep

    def log_prior_partial(self, clq, sep):
        lp = self.log_potential(clq) - self.log_potential(sep)
        return lp
        
    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        new = self.log_prior(new_cliques, new_separators)
        old = self.log_prior(old_cliques, old_separators)
        return new - old


class UniformJTDistribution(SequentialJTDistribution):
    """ A sequential formulation of P(T) = P(T|G)P(G), where
        P(G)=1/(#decomopsable graphs)
        and
        P(T|G) = 1/(#junction trees for G).
    """
    def __init__(self, p):
        self.p = p

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        return 0.0

    def log_likelihood_partial(self, cliques, separators):
        return 0.0

    def log_likelihood(self, jt):
        return 0.0




class CondUniformJTDistribution(SequentialJTDistribution):
    """ A sequential formulation of P(T) = P(T, G) = P(T|G)P(G), where
        P(G)=1/(#decomopsable graphs)
        and
        P(T|G) = 1/(#junction trees for G).
    """
    def __init__(self, p):
        self.p = p

    def ll(self, graph):
        pass

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        return -parallelDG.graph.junction_tree.log_n_junction_trees_update_ratio(new_separators,
                                                                               old_JT, new_JT)

class CondUniformGivenSizeJTDistribution(SequentialJTDistribution):
    """ A sequential formulation of P(T) = P(T, G) = P(T|G)P(G), where
        P(G)=1/(#decomopsable graphs) * I(size of G = k)
        and
        P(T|G) = 1/(#junction trees for G).
    """
    def __init__(self, p, size):
        self.p = p
        self.size = size

    def ll(self, graph):
        pass

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        graph = new_JT.to_graph()
        #print("order: " + str(graph.order()) + " size: " + str(graph.size()))
        if graph.size() <= self.size:
            #print("ok")
            return -parallelDG.graph.junction_tree.log_n_junction_trees_update_ratio(new_separators,
                                                                                   old_JT, new_JT)
        else:
            #print("bad size")
            return -np.inf


class LogLinearJTPosterior(SequentialJTDistribution):
    """
    Posterior for a log-linear model.
    """

    def init_model(self, X, cell_alpha, levels, cache_complete_set_prob={},
                   counts={}):
        """
        Args:
            cell_alpha: the constant number of pseudo counts for each cell
            in the full distribution.
        """
        self.p = len(levels)
        self.levels = levels
        self.cache_complete_set_prob = cache_complete_set_prob
        self.cell_alpha = cell_alpha
        self.data = X
        self.no_levels = np.array([len(l) for l in levels])
        self.counts = counts

    def init_model_from_json(self, sd_json):
        self.init_model(np.array(sd_json["data"]), # TODO: Might be a bug
                        sd_json["parameters"]["cell_alpha"],
                        np.array(sd_json["parameters"]["levels"]),
                        cache_complete_set_prob={})

    def get_json_model(self):
        return {"name": "loglin_jt_post",
                "parameters": {"cell_alpha": self.cell_alpha,
                               "levels": [list(l) for l in self.levels]},
                "data": self.data.tolist()}

    def log_likelihood(self, tree):
        separators = jtlib.separators(tree)
        return loglin.log_likelihood_partial(tree.nodes(),
                                             separators,
                                             self.no_levels,
                                             self.cell_alpha,
                                             self.counts,
                                             self.data,
                                             self.levels,
                                             self.cache_complete_set_prob)

    def log_likelihood_partial(self, cliques, separators):
        return loglin.log_likelihood_partial(cliques,
                                             separators,
                                             self.no_levels,
                                             self.cell_alpha,
                                             self.counts,
                                             self.data,
                                             self.levels,
                                             self.cache_complete_set_prob)

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        log_mu_ratio = parallelDG.graph.junction_tree.log_n_junction_trees_update_ratio(new_separators,
                                                                                      old_JT, new_JT)
        ll_ratio = self.log_likelihood_diff(old_cliques,
                                            old_separators,
                                            new_cliques,
                                            new_separators,
                                            old_JT,
                                            new_JT)
        return ll_ratio - log_mu_ratio

    def log_likelihood_diff(self, old_cliques, old_separators,
                            new_cliques, new_separators, old_JT, new_JT):
        """ Log-likelihood difference when cliques and separators are added and
            removed.
        """
        old = loglin.log_likelihood_partial(old_cliques,
                                            old_separators,
                                            self.no_levels,
                                            self.cell_alpha,
                                            self.counts,
                                            self.data,
                                            self.levels,
                                            self.cache_complete_set_prob)
        new = loglin.log_likelihood_partial(new_cliques,
                                            new_separators,
                                            self.no_levels,
                                            self.cell_alpha,
                                            self.counts,
                                            self.data,
                                            self.levels,
                                            self.cache_complete_set_prob)
        return new - old

    def __str__(self):
        return "loglin_posterior_n_"+str(self.data.shape[1])+"_p_"+str(self.p)+"_pseudo_obs_"+str(self.cell_alpha)


class GGMJTPosterior(SequentialJTDistribution):
    """ Posterior of Junction tree for a GGM.
    """
    def init_model(self, X, D, delta, cache={}):
        self.parameters = {"delta": delta,
                           "D": D}
        self.SS = X.T * X
        self.X = X
        self.cache = cache

        self.n = X.shape[0]
        self.p = X.shape[1]
        self.idmatrices = [np.identity(i) for i in range(self.p+1)]

    def init_model_from_json(self, sd_json):
        self.init_model(np.asmatrix(sd_json["data"]),
                        np.asmatrix(sd_json["parameters"]["D"]),
                        sd_json["parameters"]["delta"],
                        {})

    def get_json_model(self):

        return {"name": "ggm_jt_post",
                "parameters": {"delta": self.parameters["delta"],
                               "D": self.parameters["D"].tolist()},
                "data": self.X.tolist()}

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        log_mu_ratio = parallelDG.graph.junction_tree.log_n_junction_trees_update_ratio(new_separators,
                                                                                      old_JT, new_JT)
        log_J_ratio = self.ll_diff(old_cliques,
                                   old_separators,
                                   new_cliques,
                                   new_separators,
                                   old_JT,
                                   new_JT)
        return log_J_ratio - log_mu_ratio

    def ll_diff(self,
                old_cliques,
                old_separators,
                new_cliques,
                new_separators,
                old_JT,
                new_JT):
        old = gaussian_graphical_model.log_likelihood_partial(self.SS, self.n,
                                                              self.parameters["D"],
                                                              self.parameters["delta"],
                                                              old_cliques,
                                                              old_separators,
                                                              self.cache,
                                                              self.idmatrices)

        new = gaussian_graphical_model.log_likelihood_partial(self.SS, self.n,
                                                              self.parameters["D"],
                                                              self.parameters["delta"],
                                                              new_cliques,
                                                              new_separators,
                                                              self.cache,
                                                              self.idmatrices)

        return new - old

    def log_likelihood(self, tree):
        return gaussian_graphical_model.log_likelihood(tree, self.SS, self.n,
                                                       self.parameters["D"],
                                                       self.parameters["delta"],
                                                       self.cache)

    def log_likelihood_partial(self, cliques, separators):
        return gaussian_graphical_model.log_likelihood_partial(self.SS, self.n,
                                                               self.parameters["D"],
                                                               self.parameters["delta"],
                                                               cliques, separators, self.cache)

    def __str__(self):
        return "ggm_posterior_n_" + str(self.n) + "_p_" + str(self.p) + "_prior_scale_" + str(
            self.parameters["delta"]) + "_shape_x"
