import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hBeta import PThBeta
import seaborn as sns
import pathlib
import collections
from utils.utils_func import load_database, _inv_transform, sample_pca_residuals_distribution, save_xls
from preprocessing import epi_year_cases_matrix
import matplotlib.lines as mlines
from collections import defaultdict
from utils.utils_func import _sim_square_pos
from lalonde import _load_lalonde
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DIR_LAN = 'C:\\Users\\itaym\\Google Drive\\University - Itay\\Msc\\Thesis\\Epi\\Results\\Lanlonde\\X\\'
DIR_SIM = 'C:\\Users\\itaym\\Google Drive\\University - Itay\\Msc\\Thesis\\Epi\\Results\\square_pos\\'


def _load_pickle(file=('3_9_1_org'), data='X_lalonde'):
    dir = DIR_LAN if data == 'X_lalonde' else DIR_SIM
    file = dir + data + '_' + file + '.pickle'
    with (open(file, "rb")) as openfile:
        d_res, d_info = pickle.load(openfile)
    return d_res, d_info


def _dyadic_cube_to_samples(X, pt):
    min_dyc = pt.arr_med - pt.diff_vec / 2
    max_dyc = pt.arr_med + pt.diff_vec / 2
    int_mat = pt.arr_ind @ pt.weight_vec
    d = dict.fromkeys(int_mat, 0)
    if X.shape[1] == 2:
        for pt_cell in d:
            d[pt_cell] = np.where(((X.iloc[:, 0] < max_dyc[pt_cell, 0]) & (X.iloc[:, 1] < max_dyc[pt_cell, 1])) &
                                  ((X.iloc[:, 0] > min_dyc[pt_cell, 0]) & (X.iloc[:, 1] > min_dyc[pt_cell, 1])))[0]
    else:
        for pt_cell in d:
            d[pt_cell] = np.where(((X.iloc[:, 0] < max_dyc[pt_cell, 0]) & (X.iloc[:, 1] < max_dyc[pt_cell, 1]) &
                                   (X.iloc[:, 2] < max_dyc[pt_cell, 2])) &
                                  ((X.iloc[:, 0] > min_dyc[pt_cell, 0]) & (X.iloc[:, 1] > min_dyc[pt_cell, 1]) &
                                   (X.iloc[:, 2] > min_dyc[pt_cell, 2])))[0]
    return d


def _dyadic_prediction(pred, dyadic_sample):
    d_pred = dict.fromkeys(dyadic_sample.keys(), None)
    for pt_cell in dyadic_sample:
        pred_sample = pred[dyadic_sample[pt_cell]]
        if pred_sample.shape[0] == 0:
            pass
        else:
            d_pred[pt_cell] = np.mean(pred_sample)
    return d_pred
    # s_obs = np.sum(((X.iloc[i, 0] < max_dyc[:, 0]) & (X.iloc[i, 1] < max_dyc[:, 1])) & (
    #             (X.iloc[i, 0] > min_dyc[:, 0]) & (X.iloc[i, 1] > min_dyc[:, 1])))

dataset = 'square_pos'#'X_lalonde'
if dataset == 'X_lalonde':
    X, y, a = _load_lalonde()
else:
    X, a = _sim_square_pos()

data_plot = pd.DataFrame({'x1': X.iloc[:, 0], 'x2': X.iloc[:, 1], 'a': a})

d_res, d_info = _load_pickle()
# d_res, d_info = _load_pickle(file=('2_10_1_org'), data='square_pos')

tr_prop, cr_prop = np.mean(a), 1-np.mean(a)
df = pd.DataFrame({'control': d_res['control']['pi'], 'treated': d_res['treated']['pi']})
df_sum_y = np.sum(df, axis=1)
propensity = (df['treated'] * tr_prop) / (df['control'] * cr_prop + df['treated'] * tr_prop)

pt = PThBeta(seg_1dim=5)
pt.set_int_coords(data=X, gamma=0.2, sup_01=False)
int_mat = pt.arr_ind @ pt.weight_vec
u = np.zeros((pt.intervals_1dim, pt.intervals_1dim))
dyadic_sample = _dyadic_cube_to_samples(X, pt)
ratio_dens = df['treated'] / (df['treated'] + df['control'])

k = 0
sns.jointplot(data=data_plot, x='x1', y='x2', hue='a', s=10)
fig, ax = plt.subplots(ncols=3, nrows=1)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for i in range(pt.intervals_1dim):
    for j in range(pt.intervals_1dim - 1, -1, -1):
#    for j in range(pt.intervals_1dim):
        if dyadic_sample[k].shape[0] > 0:
            u[pt.arr_ind[i], pt.arr_ind[j]] = propensity[k]#d_res['treated']['pi'][k]
        else:
            u[pt.arr_ind[i], pt.arr_ind[j]] = None
        k += 1
sns.heatmap(u, cmap='viridis', ax=ax[0], xticklabels=False, yticklabels=False, cbar=True, cbar_ax=cbar_ax)
ax[0].set_title('P(A=1|X) - PThBeta')

# lr
clf = LogisticRegression()
clf.fit(X, a)
pred_rf = clf.predict_proba(X)[:, 1]
d_pred_rf = _dyadic_prediction(pred_rf, dyadic_sample)
u = np.zeros((pt.intervals_1dim, pt.intervals_1dim))
k = 0
for i in range(pt.intervals_1dim):
    # for j in range(pt.intervals_1dim):
    for j in range(pt.intervals_1dim - 1, -1, -1):
        if d_pred_rf[k] is not None:
            u[pt.arr_ind[i], pt.arr_ind[j]] = d_pred_rf[k]
        else:
            u[pt.arr_ind[i], pt.arr_ind[j]] = None
        k += 1
sns.heatmap(u, cmap='viridis', ax=ax[1], xticklabels=False, yticklabels=False, cbar=False)
ax[1].set_title('P(A=1|X) - Logistic Regression')

X['x3'] = X.iloc[:, 0] * X.iloc[:, 1]
clf = LogisticRegression()
clf.fit(X, a)
pred_rf = clf.predict_proba(X)[:, 1]
d_pred_rf = _dyadic_prediction(pred_rf, dyadic_sample)
u = np.zeros((pt.intervals_1dim, pt.intervals_1dim))
k = 0
for i in range(pt.intervals_1dim):
    # for j in range(pt.intervals_1dim):
    for j in range(pt.intervals_1dim - 1, -1, -1):
        if d_pred_rf[k] is not None:
            u[pt.arr_ind[i], pt.arr_ind[j]] = d_pred_rf[k]
        else:
            u[pt.arr_ind[i], pt.arr_ind[j]] = None
        k += 1
sns.heatmap(u, cmap='viridis', ax=ax[2], xticklabels=False, yticklabels=False, cbar=False)
ax[2].set_title('P(A=1|X) - Logistic Regression\nwith interaction element')
fig.tight_layout(rect=[0, 0, .9, 1])


# RF
plt.subplots()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, a)
pred_rf = clf.predict_proba(X)[:, 1]
d_pred_rf = _dyadic_prediction(pred_rf, dyadic_sample)
u = np.zeros((pt.intervals_1dim, pt.intervals_1dim))
k = 0
for i in range(pt.intervals_1dim):
    for j in range(pt.intervals_1dim):
        if d_pred_rf[k] is not None:
            u[pt.arr_ind[i], pt.arr_ind[j]] = d_pred_rf[k]
        else:
            u[pt.arr_ind[i], pt.arr_ind[j]] = None
        k += 1
sns.heatmap(u, cmap='viridis')



plt.subplots()
plt.plot(d_res['control']['pi'], label=['control'])
plt.plot(d_res['treated']['pi'], label=['treated'])

# plt.plot(d_g_hbeta['control']['pi'], label=['control'])
# plt.plot(d_g_hbeta['treated']['pi'], label=['treated'])

plt.subplots()
plt.plot(np.arange(propensity.shape[0]), propensity.sort_values(), label='propensity score')
plt.scatter(x = propensity.sort_values().index, y = d_res['control']['pi'][propensity.sort_values().index], s=3)
plt.scatter(x = propensity.sort_values().index, y = d_res['treated']['pi'][propensity.sort_values().index], s=3)


plt.subplots()
propensity_RCT = (df['treated'] * 0.5) / (df['control'] * 0.5 + df['treated'] * 0.5)
plt.plot(propensity_RCT.sort_values().index, propensity_RCT.sort_values(), label='propensity RCT score')
plt.plot(np.arange(propensity.shape[0]), propensity.sort_values(), label='propensity score')


dx = df.loc[df.sum(axis=1) > 0.01]
df_dx = pd.concat([dx, propensity.loc[dx.index], propensity_RCT.loc[dx.index]], axis=1)

plt.subplots()
# plt.scatter(df_dx.index, df_dx.iloc[:, 3], label='propensity RCT score')
plt.scatter(df_dx.reset_index().index, df_dx.iloc[:, 2], label='propensity score')
plt.scatter(df_dx.reset_index().index, df_dx.iloc[:, 1], label='density treated')
plt.scatter(df_dx.reset_index().index, df_dx.iloc[:, 0], label='density control')


plt.subplots()
plt.plot(df_dx.reset_index().index, df_dx.iloc[:, 3], label='propensity RCT score')
plt.plot(df_dx.reset_index().index, df_dx.iloc[:, 2], label='propensity score')
plt.plot(df_dx.reset_index().index, df_dx.iloc[:, 1], label='density treated')
plt.plot(df_dx.reset_index().index, df_dx.iloc[:, 0], label='density control')
plt.legend()

plt.subplots()
plt.scatter(x=propensity_RCT.index, y =propensity_RCT, label='propensity RCT score')
plt.scatter(x=propensity.index, y =propensity, label='propensity score')
plt.plot(propensity.index, d_res['control']['pi'], label='density control', c='r')
plt.plot(propensity.index, d_res['treated']['pi'], label='density treated', c='g')
plt.legend()


### FOR LANLODE
# dyadic_sample = _dyadic_cube_to_samples(pca_full, pt)
# for key in np.arange(512):
#     if dyadic_sample[key].shape[0] == 0:
#         dyadic_sample.pop(key, None)
# for key in dyadic_sample:
#     d_conts[key] = dyadic_sample[key].shape[0]

# df_selected = pd.DataFrame({'index':d_conts.keys(), 'counts':d_conts.values(),'p_treated':propensity.loc[dyadic_sample.keys()]})
# plt.scatter(x=np.log(df_selected['counts']), y=df_selected['p_treated'])
# plt.title('P(A=1|X) vs. counts of observations in log scale \n(per dyadic cube)')