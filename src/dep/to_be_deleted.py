# """
# I tried different ways of segmentations, but it gives the same results... doesn't really matter
# """
#
# import numpy as np
# import collections
# import pandas as pd
# import itertools
# from scipy.stats import betabinom
# import matplotlib.pyplot as plt
# from utils.timer import Timer
#
#
# def Tree():
#     """create generic tree"""
#     return collections.defaultdict(Tree)
#
#
# def _segmentation(arr, ax):
#     """
#     Binary segementation according one axis
#     Args:
#         arr (np.array): data
#         ax (int): ax to use for splitting the space
#
#     Returns:
#         p, q - right and left node
#     """
#
#     if isinstance(arr, np.ndarray):
#         mid_point = np.median(arr[:, ax])
#         p = arr[arr[:, ax] <= mid_point, :]
#         q = arr[arr[:, ax] > mid_point, :]
#     # else:
#     #     mid_point = np.median(arr['p_ind'][:,ax])
#     #     p = arr[arr[:, ax] <= mid_point, :]
#     #     q = arr[arr[:, ax] > mid_point, :]
#     # todo:
#     #   new version with p_ind and q_ind
#     return p, q
#
#
# def perm_org(l):
#     return list(set(itertools.permutations(l, len(l))))
#
# def perm_ver1(l):
#     s = set(itertools.permutations(l, len(l)))
#     return list(s)
#
# def perm_ver2(l):
#     s = set(itertools.permutations(l, len(l)))
#     return [*s, ]
#
# order = {0: ['1', '0', '0', '1'], 1: ['0','1','2'], 2: ['0','1','2','3','0','1','2','3', '0','1','2','3'],
#          3:['0','1','2','3','4', '0','1','2','3', '4', '0', '1']} #, '0','1','2','3', '4'
# timer = Timer()
# for i in range(4):
#     timer.start()
#     perm_org(order[i])
#     timer.stop(process="perm_org")
#     timer.start()
#     k1= perm_ver1(order[i])
#     timer.stop(process="perm_ver1")
#     # timer.start()
#     # k2= perm_ver2(order[i])
#     # timer.stop(process="perm_ver2")


### Cython training
"""
I tried to use cyhton in my code
"""
def test(x):
    y = 0
    for i in range(x):
        y += 1
    return y

