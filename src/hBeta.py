"""

"""
#   TODO (12/11/2020):
#       * set_int_coords - problem with mapping interval and bound_support
#       * I shall fix the code in cases where bound_support is not 01 -- works well for Dani's code

import numpy as np
import collections
import pandas as pd
import itertools
from scipy.stats import betabinom
import matplotlib.pyplot as plt
from utils.timer import Timer


def Tree():
    """create generic tree"""
    return collections.defaultdict(Tree)


def _segmentation(arr, ax):
    """
    Binary segementation according one axis
    Args:
        arr (np.array): data
        ax (int): ax to use for splitting the space

    Returns:
        p, q - right and left node
    """

    if isinstance(arr, np.ndarray):
        mid_point = np.median(arr[:, ax])
        p = arr[arr[:, ax] <= mid_point, :]
        q = arr[arr[:, ax] > mid_point, :]
    # else:
    #     mid_point = np.median(arr['p_ind'][:,ax])
    #     p = arr[arr[:, ax] <= mid_point, :]
    #     q = arr[arr[:, ax] > mid_point, :]
    # todo:
    #   new version with p_ind and q_ind
    return p, q


def perm(l):
    """
    calculate the number of optional permutations
    Args:
        l:
    Returns:
        list of tuples
    """
    return list(set(itertools.permutations(l, len(l))))


def perm_matrix(p, seg_one_var):
    """
    Create a matrix of indices
        for example (p=2, seg_one_var=2):
        array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]], dtypes=int8)
    Args:
        p (int): dimension
        seg_one_var (int): number of segmentation for each dimension

    Returns:
        array with size of (p^seg_one_var, p)
    """
    empty_mat = np.empty((seg_one_var ** p, p))
    for i in range(p):
        vec = np.repeat(np.arange(seg_one_var, dtype=int), seg_one_var ** (p - i - 1))
        if vec.shape[0] != seg_one_var ** p:
            vec = np.tile(vec, seg_one_var ** i)
        empty_mat[:, i] = vec
    return empty_mat.astype(np.int8, copy=False)
    # return np.fliplr(empty_mat.astype(np.int8, copy=False))

def med_vec_array(dict_of_seg, arr_ind):
    """
    return arrays of mediaan and arrays of vec
    Note that I am starting from zero of the arr_vec, but from +1 of med (cutting the top_
    Args:
        dict_of_seg:
        arr_ind:

    Returns:

    """
    a_med = np.empty(shape=arr_ind.shape)
    a_vec = np.empty(shape=arr_ind.shape)
    for ind, row in enumerate(arr_ind):
        a_med[ind, :] = np.array([dict_of_seg[p]['med_vec'][row[p]] for p in range(arr_ind.shape[1])])
        # a_med[ind, :] = np.array([dict_of_seg[p]['med_vec'][row[p]+1] for p in range(arr_ind.shape[1])])
        a_vec[ind, :] = np.array([dict_of_seg[p]['vec'][row[p] + 1] for p in range(arr_ind.shape[1])])
        # a_vec[ind, :] = np.array([dict_of_seg[p]['vec'][row[p]] for p in range(arr_ind.shape[1])])
    return a_med, a_vec


class PThBeta:

    def __init__(self, seg_1dim=2):
        """

        Args:
            seg_1dim (int): times to segment each variable
                        (2*seg_one_var is equal to the number of interval at each variable at its L level)
        """
        self.seg_1dim = seg_1dim
        self.intervals_1dim = 2 ** self.seg_1dim
        self.arr_ind = None
        self.arr_med = None
        self.arr_vec = None
        self.diff_vec = None

    def _init_params(self, p):
        """
        by given a dataset, init parameters -
            L - level of tree
            I - number of intervals in the deepest level L
        Args:
            p(int): number of variables in a given dataset
        """
        self.L = int(p * self.seg_1dim)  # level of the tree
        self.I = 2 ** self.L  # number of interval in L-level
        # self.weight_vec = np.array([self.intervals_1dim ** i for i in range(p)]) #np.array([4, 1])
        self.weight_vec = np.flip(np.array([self.intervals_1dim ** i for i in range(p)])) #np.array([4, 1])

    # def _ind_tree(self):
    #     """create indices trees"""
    #     # todo:
    #     #   NOT SURE if relevant
    #     t = Tree()
    #     for l in range(1, self.L + 1):
    #         for node in range(1, 2 ** (l - 1) + 1):
    #             p_ind = np.arange(1, 2 ** (self.L - l) + 1) + (node - 1) * 2 ** (self.L - l + 1)
    #             q_ind = p_ind + 2 ** (self.L - l)
    #             t[l][node] = {"p_ind": p_ind, "q_ind": q_ind}
    #
    #             # t[1][1] = _segmentation_p(arr, 0)
    #             # t[2][1] = {"p": _segmentation_p(t[1][1]['p'], 1), "q": _segmentation_q(t[1][1]['q'], 1)}
    #     return t

    def _ind_matrix(self):
        """
        create incidence matrix to easily compute the pathway for each node
        """
        prod_ind_mat = np.empty((self.L, self.I))
        for i, l in enumerate(range(1, self.L + 1)):
            p_ind = np.arange(2 ** (l - 1), 2 ** l, dtype=int) if i > 0 else np.array([1])
            q_ind = p_ind + 2 ** self.L - 1
            p_q = np.column_stack((p_ind, q_ind)).flatten()
            prod_ind_mat[l - 1, :] = np.repeat(p_q, 2 ** (self.L - l))

        # the matrix reduce by 1, due to pythonic syntax
        ret_prod_ind = prod_ind_mat - 1
        return ret_prod_ind.astype(int)

    def _bound_suppurt(self, data, gamma, sup_01=False):
        """
        Args:
            data (pd.DataFrame):
            sup_01 (bool): if True the support is bounded to be [0,1],
                            otherwise the support is defined by the data and variable gamma
            gamma (float): determine to which percentage to extend the space at each dimension

        Returns:
            dict_of_seg (dict): nested dictionary that include information of how the space is divided
        """
        dict_of_seg = collections.defaultdict(dict)
        p = data.shape[1]
        if sup_01:
            for variable in range(p):
                vec = np.linspace(0, 1, self.intervals_1dim + 1)
                vec_diff = vec[1] - vec[0]
                vec_med = vec[1:] - vec_diff / 2  # vec[1:] - vec_diff / 2
                dict_of_seg[variable] = {"vec": vec, "med_vec": vec_med, 'diff': vec_diff,
                                         'min': 0, 'max': 1}
        else:
            for ind, col in enumerate(data):
                min_val, max_val = np.min(data.loc[:, col]), np.max(data.loc[:, col])
                _range = max_val - min_val
                vec = np.linspace(min_val - gamma * _range, max_val + gamma * _range, self.intervals_1dim + 1)
                vec_diff = vec[1] - vec[0]
                vec_med = vec[1:] - vec_diff / 2
                dict_of_seg[ind] = {"vec": vec, "med_vec": vec_med, 'diff': vec_diff,
                                    'min': min_val - np.abs(gamma * min_val), 'max': max_val + gamma * max_val}
        return dict_of_seg

    def _map_u(self, u):
        """
        mapping for each point (represented as vector in size p)  the index of the appropriate cell
        Args:
            u: one point observation, in size [1 x p]

        Returns:
        """
        # todo:
        #   utils func - check that its not out of bounds --> problem with the support
        p = self.arr_vec.shape[1]
        mat_index = np.where(np.sum(self.arr_vec > u, 1) == p)[0][0]
        return self.arr_ind[mat_index, :] @ self.weight_vec

    def plot_sampler_grid(self, data, pred_sample, title=None):
        """
        For 2 or 3 dimensions display the grid and the predictive distribution samples
        Args:
            data:
            pred_sample:
            title (None | str):

        Returns:

        """
        # todo:
        #   utils - check data and pred_sample -- same shape[1]
        fig = plt.figure()
        min_vals = self.arr_med[0, :]
        p = min_vals.shape[0]
        int_mat = self.arr_ind @ self.weight_vec + 1
        np_d = np.array(data)
        ind = 0
        if p == 2:
            ax = plt.axes()
            for i in range(self.intervals_1dim):
                for j in range(self.intervals_1dim):
                    # ax.text(min_vals[0] + i * self.diff_vec[0], min_vals[1] + j * self.diff_vec[1],
                    #         str(int_mat[ind]), va='center', ha='center')
                    ind += 1
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xticks(pd.unique(self.arr_vec[:, 0])[:-1])
            ax.set_yticks(pd.unique(self.arr_vec[:, 1])[:-1])
            ax.set_xlim(min_vals[0] - 1.2 * (self.diff_vec[0]/2), 1.02 * (self.arr_vec[-1, 0]))
            ax.set_ylim(min_vals[1] - 1.2 * (self.diff_vec[1]/2), 1.02 * (self.arr_vec[-1, 1]))
            ax.scatter(x=pred_sample[:, 0], y=pred_sample[:, 1], alpha=0.08, s=15, c='blue')
            ax.scatter(x=np_d[:, 0], y=np_d[:, 1], alpha=0.6, s=20, c='red')
            ax.set_title(title)
            ax.grid()
        elif min_vals.shape[0] == 3:
            ax = plt.axes(projection='3d')
            for i in range(self.intervals_1dim):
                for j in range(self.intervals_1dim):
                    for k in range(self.intervals_1dim):
                        # todo: make the font size smaller
                        # ax.text(min_vals[0] + i * self.diff_vec[0], min_vals[1] + j * self.diff_vec[1],
                        #         min_vals[2] + k * self.diff_vec[2], str(int_mat[ind]), va='center', ha='center')
                        ind += 1
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xticks(pd.unique(self.arr_vec[:, 0])[:-1])
            ax.set_yticks(pd.unique(self.arr_vec[:, 1])[:-1])
            ax.set_zticks(pd.unique(self.arr_vec[:, 2])[:-1])
            ax.set_xlim(min_vals[0] - 1.2 * (self.diff_vec[0]/2), 1.02 * (self.arr_vec[-1, 0]))
            ax.set_ylim(min_vals[1] - 1.2 * (self.diff_vec[1]/2), 1.02 * (self.arr_vec[-1, 1]))
            ax.set_zlim(min_vals[2] - 1.2 * (self.diff_vec[2]/2), 1.02 * (self.arr_vec[-1, 2]))
            ax.scatter3D(pred_sample[:, 0], pred_sample[:, 1], pred_sample[:, 2], alpha=0.08, s=15, c='blue')
            ax.scatter3D(np_d[:, 0], np_d[:, 1], np_d[:, 2], alpha=0.6, s=20, c='red')
            ax.set_title(title)
            ax.grid()

        else:
            assert "this dimension cannot be displayed"

    def set_int_coords(self, data, gamma, sup_01, plot=False):
        """
        A preliminary code to define:
         - the parameters of the model, such as L - Level of the tree and by number of segmentation and deepest level
         - bound the support
         - divide the
        Args:
            data:
            sup_01:
            plot:

        Returns:

        """
        p = data.shape[1]  # number of covariates
        self._init_params(p)

        dict_of_seg = self._bound_suppurt(data, gamma=gamma, sup_01=sup_01)
        arr_ind = perm_matrix(p, self.intervals_1dim)
        arr_med, arr_vec = med_vec_array(dict_of_seg, arr_ind)
        self.arr_ind = arr_ind
        self.arr_med = arr_med
        self.arr_vec = arr_vec
        self.diff_vec = np.array([dict_of_seg[dim]['diff'] for dim in range(p)])

    def segment_trees(self, order, p):
        """
        Needs to incorporate it to the model, but the idea is as follows:
        - input:
            * order of axis for partitions (e.g [0,0,1,2])
            * dimensions, in size (int)
            * number of segementations per dimension (self.root I)
        - output:
            * indiex of the segmented
        Args:
            order:

        Returns:

        """
        t = Tree()
        t_ind = Tree()

        for ind, l in enumerate(range(self.L + 1)):
            for node in range(1, 2 ** (l) + 1, 2):
                if l == 0:
                    t[l][1] = self.arr_ind
                else:
                    prev_node = int(np.ceil(node / 2))
                    p_ind, q_ind = _segmentation(t[l - 1][prev_node], order[ind - 1])
                    t[l][node] = p_ind
                    t[l][1 + node] = q_ind

                    t_ind[l][int(np.ceil(node / 2))]['p_ind'] = p_ind @ self.weight_vec + 1
                    t_ind[l][int(np.ceil(node / 2))]['q_ind'] = q_ind @ self.weight_vec + 1

        # TODO:
        #   NEED TO FIND MORE ELEGANT SOLUTION
        map_interval = np.ones(self.I, dtype=int)
        for i in t[self.L].keys():
            map_interval[t[self.L][i][0] @ self.weight_vec] = int(i)

        return map_interval, t_ind

    def segment_trees_all_options(self, p):

        d_partitions_int = collections.defaultdict(dict)
        d_partitions_trees = collections.defaultdict(dict)

        l = []
        for i in range(p):
            l.append([i] * int(self.L / p))
        flat_list = list(itertools.chain(*l))
        orders = perm(flat_list)

        for part_num, order in enumerate(orders):
            d_partitions_int['partition_' + str(part_num + 1)], d_partitions_trees['partition_' + str(part_num + 1)] \
                = self.segment_trees(order, p=p)
        return d_partitions_int, d_partitions_trees

    def pi_hBeta_sampler(self, data, n_pts=2, a_0=1, gamma=0.05, sup_01=True, init_space=True):

        # todo:
        #   This code should run as multiprocessing (NOT NECSSERAILY)
        timer = Timer()

        np_d = np.array(data)
        p = np_d.shape[1]
        m = np_d.shape[0]

        # divide the space - crucial to initialization
        timer.start()
        if init_space:
            self.set_int_coords(data=data, gamma=gamma, sup_01=sup_01, plot=True)

        interval_map = np.array([self._map_u(np_d[i, :]) for i in range(m)])
        freq = pd.Series({x + 1: np.sum(interval_map == x) for x in range(self.I)})

        d2, d2_trees = self.segment_trees_all_options(p)
        d2_np = np.array(pd.DataFrame(d2))      # hbeta partitions

        print(20 * '*' + '\np={0}, Level={1}, intervals at level-L={2}\nnumber of differet segmentations={3}'
              .format(p, self.L, self.I, d2_np.shape[1]))

        n_parts = d2_np.shape[1]  # number of partitions
        phi_sample = np.empty(shape=(2 * (self.I - 1), n_pts, n_parts), dtype=np.float32)
        log_node_dist_mat = np.empty((self.I - 1, n_parts), dtype=np.float32)
        timer.stop(process="set_int_coords")

        # For each segmentation permutation, generate s (2) posterior samples and compute posterior distribution
        # For posterior samples we need to generate (I-1) * n.pst * n.perm samples of phi

        # todo:
        #   this part here need to be computed as multiprocessing
        node_ind = 0
        timer.start()
        for ind, l in enumerate(range(1, self.L + 1)):
            for node in range(1, 2 ** (l - 1) + 1):

                # node_p = [np.sum(freq[d2_trees['partition_' + str(part)][l][node]]) for part in range(1, 7)]
                # # pd.DataFrame([d2_trees['partition_' + str(part)][l][node] for part in range(1, 7)])
                # node_q = [np.sum(freq[d2_trees['partition_' + str(part)][l][node+1]]) for part in range(1, 7)]
                # # pd.DataFrame([d2_trees['partition_' + str(part)][l][node+1] for part in range(1, 7)])

                # todo:
                #   *** I checked this part and I have got the same results as in Dani's part ****
                node_p = [np.sum(freq[d2_trees['partition_' + str(part)][l][node]['p_ind']])
                          for part in range(1, 1 + n_parts)]
                # pd.DataFrame([d2_trees['partition_' + str(part)][l][node] for part in range(1, 7)])
                node_q = [np.sum(freq[d2_trees['partition_' + str(part)][l][node]['q_ind']])
                          for part in range(1, 1 + n_parts)]

                # # todo:
                # #   enable here a utils check function
                # print('level:{}, \nnode p:{}, node q:{}'.format(l, node_p, node_q))
                # print('level:{}, \n sum_node_part:{},'.format(l, np.array(node_p) + np.array(node_q)))

                # this part is equal to phi_(l,node) ~ Beta(a0 + N_(l,2*node - 1), a0 + N_(l,2*node))
                phi_sample[node_ind, :, :] = \
                    np.random.beta(node_p + np.array(a_0), node_q + np.array(a_0), size=(n_pts, n_parts))
                # todo:
                #   I used newer and faster sampler for Beta ^^
                #     np.array([np.random.beta(node_p[i] + a_0, node_q[i] + a_0, size=n_pts)
                #               for i in range(n_parts)]).T
                phi_sample[node_ind + self.I - 1, :, :] = 1 - phi_sample[node_ind, :, :]

                #  compute Beta-Binomial N_(l,2*j-1)|N_(l-1,j) ~ Beta-Binomial(N_(l-1,j), a0, a0)
                if a_0 == 0:
                    log_node_dist_mat[node_ind, :] = -np.log(1 + node_p + node_q)
                else:
                    log_node_dist_mat[node_ind, :] = \
                        betabinom.logpmf(np.array(node_p), np.array(node_p) + np.array(node_q), a_0, a_0)
                node_ind += 1

        timer.stop(process="sampling loop")
        prod_ind_mat = self._ind_matrix()

        # phi_4d_mat = np.empty(shape=(self.L, self.I, n_parts, n_pts))


        pi_hBeta_sample = np.empty(shape=(self.I, n_parts, n_pts), dtype=np.float32) # i tried to change it to int
        pi_map_sample = np.empty(shape=(self.I, n_parts, n_pts), dtype=np.float32) # i tried to change it to int

        timer.start()
        # todo:
        #   this code needs to be written more efficiently - using matrix multiplication
        # todo:
        #   *** I checked this part and it seems ok ... it is hard to make further check :( ****
        # COMPARESION TO DANI'S CODE:
        #   d2_np =  Uxy.ind.mat
        #   hBeta.ind.mat ~= equiv to d2_trees

        for s in range(n_pts):
            for part in range(n_parts):
                pi_hBeta_sample[:, part, s] = np.prod(phi_sample[:, s, part][prod_ind_mat[:, :]], 0)
                pi_map_sample[:, part, s] = pi_hBeta_sample[:, part, s][d2_np[:, part] - 1]
                # print(np.sum(np.prod(phi_sample[:, s, part][prod_ind_mat[:, :]], 0)))

        timer.stop(process="pi_map_sample")

        # todo:
        #   for each node-value --> map to its corresponds index
        # phi_multiply_index =

        log_pdist = np.sum(log_node_dist_mat, axis=0)
        map_seg_pdist = np.exp(log_pdist - np.max(log_pdist))
        return pi_map_sample, map_seg_pdist, freq  ## equivelent to (list(pi.Uxy.sample,Uxy.seg.pdist,nvec.generic))
        # return pi_hBeta_sample, map_seg_pdist, freq
        # # todo:
        #   at the end it shall return -
        #   pi.Uxy.sample --> a dataframe of pi at the deepest level (e.g 16) * partitions (e.g 6) * samples (e.g 2)

    def pred_map_sample(self, pi_map_sample, map_seg_pdist, n_samples=1000):
        """
        sampling from the posterior distribution
        Args:
            pi_map_sample:
            map_seg_pdist:
            n_samples (int): number of samples

        Returns:

        """
        p = self.diff_vec.shape[0]
        n_parts, n_pts = pi_map_sample.shape[1], pi_map_sample.shape[2]  # number of partitions, number of
        pi_res = np.reshape(pi_map_sample, (self.I, n_parts * n_pts))  # equvi. to pi_Uxy_sample_2D
        # pi_res is ordered as sample, sample ... and then different segmentation
        # i.e.  pi_res[:,1] = pi_map_sample[:,0,1]
        pi_cumsum = np.cumsum(pi_res, axis=0)

        u_mat = np.repeat(np.random.uniform(0, 1, size=n_parts * n_pts)[np.newaxis, :], self.I, axis=0)
        #pred_vec = np.sum(pi_cumsum < u_mat, axis=0)    # ? todo: why i have promlem with 1993??

        # todo: I tried something a bit diffenet
        pred_vec = np.minimum(self.I-1, np.sum(pi_cumsum < u_mat, axis=0))   # ? todo: why i have promlem with 1993??

        # pred_vec_2 = self.arr_ind[pred_vec] @ self.weight_vec
        map_pred_vec = self.arr_med[pred_vec] + \
                       (np.random.uniform(0, 1, size=(n_parts * n_pts, p)) - 0.5) * (self.diff_vec)
        weighted_prob = np.repeat(map_seg_pdist, n_pts) / np.sum(np.repeat(map_seg_pdist, n_pts))
        # important sampling
        imp_samp_ind = np.random.choice(n_parts * n_pts, size=n_samples, replace=True, p=weighted_prob)
        return map_pred_vec[imp_samp_ind, :]

    def comp_pi_post_mean(self, pi_map_sample, map_seg_pdist):
        """

        Args:
            pi_map_sample:
            map_seg_pdist:

        Returns:

        """
        n_parts, n_pts = pi_map_sample.shape[1], pi_map_sample.shape[2]

        pi_res = np.reshape(pi_map_sample, (pi_map_sample.shape[0], -1))
        # todo:
        #   this is equal to:
        #       - pi_res[0, :5] = pi_map_sample[0, 0, :5] # [cubes, n_parts, n_pts]
        #   HAVE TO CHECK VERY CAREFULLY THAT THE REPEATITION IS DONE RIGHT
        # mix_weights = np.repeat(map_seg_pdist, n_pts) / sum(np.repeat(map_seg_pdist, n_pts))
        mix_weights = np.tile(map_seg_pdist, n_pts) / sum(np.tile(map_seg_pdist, n_pts))
        pi_mixture = pi_res @ mix_weights
        return pi_mixture
