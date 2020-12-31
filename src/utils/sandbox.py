from hBeta import PThBeta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr = np.array([[i, j, k, f] for i in range(10) for j in range(10) for k in range(10) for f in range(10)])

data_2d = pd.DataFrame(data={'a': np.random.normal(2, 4, (100,)), 'b': np.random.normal(0, 2, (100,))})
data_2d = pd.DataFrame(data={'a': np.random.normal(0, 0.5, (100,)), 'b': np.random.normal(0, 0.3, (100,))})
data_3d = pd.DataFrame(data={'a': np.random.normal(2, 4, (100,)), 'b': np.random.normal(0, 2, (100,)),
                             'c': np.random.normal(20, 4, (100,))})

data_mvn_check2 = pd.DataFrame(
    np.array([[0.87580503, 0.581028579], [0.09047127, 0.030183940], [0.70096576, 0.305379390],
              [0.04415130, 0.161035341], [0.06962268, 0.019312893], [0.63440124, 0.365623569],
              [0.91155936, 0.607811270], [0.48502692, 0.205662618], [0.53453569, 0.111975300],
              [0.11593157, 0.003475548], [0.90315363, 0.678985328], [0.18146609, 0.122102401],
              [0.25917076, 0.311565857], [0.03902295, 0.092016583], [0.32349762, 0.021273496],
              [0.51786914, 0.065912152], [0.23710951, 0.045258222], [0.79722136, 0.391737063],
              [0.08603979, 0.049242457], [0.70559756, 0.320829783]]))

data_mvn_check = pd.DataFrame(
    np.array([[0.31625766, 0.1322318120], [0.48644747, 0.5211970104], [0.47121982, 0.0563142267],
              [0.77820904, 0.4515930927], [0.64744094, 0.1548979408], [0.93655996, 0.5098829765],
              [0.06104041, 0.0005595901], [0.83795157, 0.2955973913], [0.65822196, 0.4352404252],
              [0.84137458, 0.4595620040], [0.87934934, 0.5440106842], [0.20526191, 0.0030164874],
              [0.70169698, 0.2197586545], [0.21269990, 0.0085359682], [0.40487008, 0.2004494528],
              [0.87885294, 0.3140807506], [0.62653184, 0.1908542459], [0.07820190, 0.0234599696],
              [0.35227361, 0.1094640563], [0.90744679, 0.5692839471]]))

# data_mvn_check = pd.DataFrame(np.random.multivariate_normal([0, -1], cov=[[1, 0.8],[0.8, 1]], size = 200))

pt = PThBeta(seg_1dim=2)
# pt.set_int_coords(data=data_mvn_check, sup_01=True, plot=True, gamma=0.05)
order = [1, 0, 0, 1]
n_pts = 1000
# T = pt.segment_trees(order, data_mvn_check.shape[1])
# d2, d2_trees = pt.segment_trees_all_options()
# d2_df = pd.DataFrame.from_dict(d2, orient='index')

pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(data_mvn_check, n_pts=n_pts, gamma=0.1, a_0=1,
                                                         sup_01=True)
pi_mixture = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist)
print("a0=1 :{}".format(pt.arr_med.T @ pi_mixture))

pt.plot_sampler_grid(data_mvn_check, pred_sample)

borders = np.linspace(0, 1, pt.I)
fig, ax = plt.subplots()
plt.plot(np.arange(pt.I) + 0.5, pi_mixture)

##### 3d #####
pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(data_3d, n_pts=n_pts, a_0=0.1,
                                                         sup_01=False)
pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist)
pt.plot_sampler_grid(data_3d, pred_sample)


# pi_mixture = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
# print("a0=0.1 :{}".format(pt.arr_med.T @ pi_mixture))
# plt.plot(np.arange(pt.I) + 0.5, pi_mixture)
# pt = PThBeta(L=6, p=3)
# pt.set_int_coords(data=data_3d, plot=True)
# d3 = pt.segment_trees_all_options()
# d3_df = pd.DataFrame.from_dict(d3, orient='index')


# some Levels for 2:
#   2, 4, 6, 8
# some Levels for 3:
#   3, 6, 9
# some Levels for 4:
#   4, 8, 12, 16

# fig, ax = plt.subplots(ncols=2)
# for i in range(5, 6):
#     for s in range(n_samples):
#         ax[0].scatter(x=np.arange(16), y= pi_map_sample[:, i, s], c='green', alpha=0.5)
# ax[0].set_ylim([-0.05, 0.7])
#
# pi_map_sample = pt.pi_hBeta_sampler(data_mvn_check, n_samples=n_samples, a_0=0.1)
#
# for i in range(5, 6):
#     for s in range(n_samples):
#         ax[1].scatter(x=np.arange(16), y= pi_map_sample[:, i, s], c='blue', alpha=0.5)
# ax[1].set_ylim([-0.05, 0.7])



# intervals = [(i, i+1) for i in range(pt.I-1)]
#
# num_intervals = len(intervals)
# for idx, (min_int, max_int) in enumerate(intervals):
#   ax.hlines(y=pi_mixture[idx], xmin=min_int, xmax=max_int, color='black')
#
#
#
from scipy.stats import multivariate_normal
mn = [0, -1]
sgma = [[1, 0.8], [0.8, 1]]  # diagonal covariance
var = multivariate_normal(mean=mn, cov=sgma)
#
# x, y = np.random.multivariate_normal(mn, sgma, 10).T