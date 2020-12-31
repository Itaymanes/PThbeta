"""
This is how the code should look like at the end
"""

from hBeta import PThBeta
import pandas as pd
from preprocessing import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load data
dir = 'C:\\Users\\itaym\\Google Drive\\University - Itay\\Msc\\Thesis\\Epi\\Data\\Competition\\Stations_org\\'

# train data (features and outcomes)
x_tr_sj = pd.read_csv(dir + 'x_tr_sj.csv')
x_tr_iq = pd.read_csv(dir + 'x_tr_iq.csv')
y_tr_sj = pd.read_csv(dir + 'y_tr_sj.csv')
y_tr_iq = pd.read_csv(dir + 'y_tr_iq.csv')

# basic visualization
#plot_epidemic_year(y_tr_sj)

# preform pca
df_epi_year = epi_year_cases_matrix(y_tr_sj)

pt = PThBeta(seg_1dim=2)
for year in df_epi_year.columns[:3]:
    y_one_year_out = y_tr_sj.loc[~(y_tr_sj['epi_year'] == year), :]
    df_PCA, pca = apply_PCA_years(y_one_year_out, n_comp=3, transform="square_root", plot=False, save=False)

    pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(df_PCA, n_pts=1000, gamma=0.1, a_0=0.1, sup_01=False)
    pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist)
    pt.plot_sampler_grid(df_PCA, pred_sample)

# without 1992, p=2
# pt = PThBeta(seg_1dim=4)
# pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(
#     df_PCA.iloc[[0, 1] + list(np.arange(3, 15)), [0, 1]], n_pts=1000, gamma=0.1, a_0=1, sup_01=False)
# pi_mixture = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
# pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist)
# pt.plot_sampler_grid(pred_sample)
# plt.scatter(x=df_PCA.iloc[:, 0],
#             y=df_PCA.iloc[:, 1], alpha=1, c='red', s=40)
# print("a0=1 :{}".format(pt.arr_med.T @ pi_mixture))

# plt.subplots()
# plt.plot(pi_mixture)
#
# plt.subplots()
# plt.plot(df_epi_year.iloc[:,2], c='black', label='true')
# plt.plot(pca.inverse_transform(df_PCA.iloc[2,:]), c='blue', label='pca')
# pt_1992 = pt.arr_med.T @ pi_mixture
# plt.plot(pca.inverse_transform(pt_1992), c='red', label='gamma=0.1')
# plt.plot(pca.inverse_transform(np.mean(pred_sample, 0)), c='green', label='pred')

# pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(
#     df_PCA.iloc[[0, 1] + list(np.arange(3, 15)), [0, 1]], n_pts=300, gamma=0.001, a_0=1, sup_01=False)
# pi_mixture = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
# pt_1992 = pt.arr_med.T @ pi_mixture
# plt.plot(pca.inverse_transform(pt_1992), c='green', label='gamma=0')

plt.legend()
# pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(
#     df_PCA.iloc[[0, 1] + list(np.arange(3, 15)), [0, 1]], n_samples=300, gamma=0.5, a_0=1, sup_01=False)
# pi_mixture = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
# print("a0=1 :{}".format(pt.arr_med.T @ pi_mixture))
# plt.plot(pi_mixture)
# ax.scatter(df_PCA.iloc[:,0], df_PCA.iloc[:,1], df_PCA.iloc[:,2], c='skyblue', s=60)

#
#
# df_pred = pd.DataFrame(pred_sample, columns=['1', '2'])
# plt.subplots()
# sns.kdeplot(data=df_pred, x="1", y="2", fill=True, thresh=0, levels=100, cmap="mako")
# plt.scatter(x = df_PCA.iloc[[0, 1] + list(np.arange(3, 15)), [0, 1]].iloc[:,0], y = df_PCA.iloc[[0, 1] + list(np.arange(3, 15)), [0, 1]].iloc[:,1], alpha=1, c='red', s = 50)