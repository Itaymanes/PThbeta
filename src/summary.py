import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import collections
from utils.utils_func import load_database, _inv_transform, sample_pca_residuals_distribution, save_xls
from preprocessing import epi_year_cases_matrix
import matplotlib.lines as mlines


def results_table(pickle_files):
    """
    Summarize results to be in a dataframe with MSE, compared to the true-values
    Args:
        pickle_files:

    Returns:

    """
    d_results = collections.defaultdict(dict)
    for file in pickle_files:
        with (open(file, "rb")) as openfile:
            d_res, d_info = pickle.load(openfile)

        for year in d_res.keys():
            d_res[year]['pred_vals'] = _inv_transform(d_res[year]['pred_vals'], d_info['transform'])
            d_results[file][year] = np.mean((d_res[year]['true_vals'].iloc[:51] - d_res[year]['pred_vals']) ** 2)
        d_results[file]['dim'] = d_info['dim']
        d_results[file]['level'] = d_info['level']
        d_results[file]['transform'] = d_info['transform']
        d_results[file]['a_0'] = d_info['a_0']
    return pd.DataFrame.from_dict(d_results, orient='index').reset_index(drop=True)


def results_table_percent(pickle_files, y, res_method='emp_week'):
    """
    Summarize results to be in a dataframe with MSE, compared to the true-values
    Args:
        pickle_files:

    Returns:

    """
    d_results = collections.defaultdict(dict)
    for file in pickle_files:
        with (open(file, "rb")) as openfile:
            d_res, d_info = pickle.load(openfile)

        for year in d_res.keys():
            if 'pca_model' in d_res[year].keys():
                df_epi_year = epi_year_cases_matrix(y.loc[y_tr.epi_year != year, :])
                res = sample_pca_residuals_distribution(df_epi_year.T, n_comp=d_info['dim'],
                                                        trans=d_info['transform'], n_pts=1000,
                                                        res_method=res_method)  # n_pts=d_info['n_pts']
                pred_values = d_res[year]['pca_model'].inverse_transform(d_res[year]['pred_sample'][:, :])
                pred_values = _inv_transform(pred_values, d_info['transform']) + res

                labelname = 'd:' + str(d_info['dim']) + ', L:' + str(d_info['level']) + ', a0:' + str(d_info['a_0']) +\
                            ', trans:' + d_info['transform']
                d_results[year][labelname] = np.mean(np.array([d_res[year]['true_vals'].iloc[:51],] * 1000) >= pred_values, axis=0)
                d_results[year]['CUMSUM' + labelname] = \
                    np.mean(np.array([np.cumsum(d_res[year]['true_vals'].iloc[:51]),] * 1000) >= np.cumsum(pred_values, axis=1), axis=0)

    list_dfs = [pd.DataFrame.from_dict(d_results[year]) for year in d_res.keys()]
    return list_dfs

def plot_sample(sim=[(2, 8, 1, 'org')], same_scale=True):
    """
    plot the expected value of the pred sample and its transformed data per year
    Args:
        sim (list of tuple): each tuple is: dim(int), level(int), a_0 (float), transform (str)
        same_scale (bool):

    Returns:
        18 plots, for each year
    """
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(14, 12))
    for ind, file in enumerate(sim):
        filename = 'sj_' + str(file[0]) + '_' + str(file[1]) + '_' + str(file[2]) + '_' + file[3] + '.pickle'
        labelname = 'dim:' + str(file[0]) + ', L:' + str(file[1]) + ', a0:' + str(file[2]) + ', trans:' + file[3]
        with (open(DIR_RES_LOO + filename, "rb")) as openfile:
            d_res, d_info = pickle.load(openfile)
        for year, ax in zip(d_res.keys(), axes.flat):
            if ind == 0:
                ax.plot(np.arange(1, 52), d_res[year]['true_vals'].iloc[:51], label='true', c='k')
            d_res[year]['pred_vals'] = _inv_transform(d_res[year]['pred_vals'], d_info['transform'])
            ax.plot(np.arange(1, 52), d_res[year]['pred_vals'], label='pred ' + labelname)
            ax.set_title(year)
            if same_scale:
                ax.set_ylim([0, 500])
    axes[4, 3].axis('off')
    axes[4, 2].axis('off')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.15))  # , loc='lower right'
    fig.tight_layout()


def plot_week_profie(sim=[(2, 10, 1, 'org', 'full')], y=None, precentile=[5, 95], plot_precentile=True,
                     res_method="None"):
    """
    plot propfile, for both LOO and FULL. The profile can be per week or as accumulated sum of weeks
    Args:
        sim:
        y:
        precentile:
        plot_precentile:
        res_method:
        year_str:

    Returns:

    """

    # legend
    post_line = mlines.Line2D([], [], color='blue', marker='_', linestyle='',
                              markersize=10, label='Posterior predictive sample')
    hbeta_precent = mlines.Line2D([], [], color='black', marker='_', linestyle='',
                                  markersize=10, label='Posterior pred percentile' + str(precentile))
    true_vals = mlines.Line2D([], [], color='red', marker='+', linestyle='',
                              markersize=10, label='Observed values')
    true_line = mlines.Line2D([], [], color='red', marker='_', linestyle='',
                              markersize=10, label='Accumulated sum')
    emp_precent = mlines.Line2D([], [], color='red', marker='_', linestyle='',
                                markersize=10, label='Emp. percentile' + str(precentile))

    # plot_week profile
    fig, axes = plt.subplots(nrows=np.int(np.ceil(len(sim) / 2)), ncols=2, figsize=(15, 10), sharey='row')
    fig2, axes2 = plt.subplots(nrows=np.int(np.ceil(len(sim) / 2)), ncols=2, figsize=(15, 10), sharey='row')
    for ax, ax2, file in zip(axes.flat, axes2.flat, sim):
        year_str = file[4]
        if year_str is not 'full':
            filename = 'sj_' + str(file[0]) + '_' + str(file[1]) + '_' + str(file[2]) + '_' + file[3] + '.pickle'
            with (open(DIR_RES_LOO + filename, "rb")) as openfile:
                d_res, d_info = pickle.load(openfile)
        else:
            filename = 'full_' + 'sj_' + str(file[0]) + '_' + str(file[1]) + '_' + str(file[2]) + '_' + file[3] + \
                       '.pickle'
            with (open(DIR_RES_FULL + filename, "rb")) as openfile:
                d_res, d_info = pickle.load(openfile)

        labelname = str(year_str) + ' - dim=' + str(file[0]) + ', L=' + str(file[1]) + ', a0=' + str(file[2]) \
                    + ', trans=' + file[3] + ', res=' + res_method
        if res_method is not "None":
            if y is None:
                raise Exception("The original signal is needed in order to compute the residuals")
            else:

                df_epi_year = epi_year_cases_matrix(y) if year_str == 'full' \
                    else epi_year_cases_matrix(y.loc[y_tr.epi_year != year_str, :])
                res = sample_pca_residuals_distribution(df_epi_year.T, n_comp=d_info['dim'],
                                                        trans=d_info['transform'], n_pts=1000,
                                                        res_method=res_method)  # n_pts=d_info['n_pts']
        else:
            res = np.zeros(shape=(1000, 51))  # np.zeros(shape=(d_info['n_pts']))
            # for year in df_epi_year.columns:
            #     ax.scatter(x=df_epi_year.index, y=df_epi_year.loc[:, year], c='red', alpha=0.18, marker='+',
            #                label='true_values')
            #### plot residuals
            # d_res = dict()
            # for dims in range(1, 5):
            #     d_res[dims] = sample_pca_residuals_distribution(df_epi_year.T, n_comp=dims,
            #                                                     trans=d_info['transform'], n_pts=d_info['n_pts'])
            #     fig, ax = plt.subplots()
            #     ax.boxplot(df_res)
            #     plt.show()
            # df_res = pd.DataFrame.from_dict(d_res)

        pred_values = d_res[year_str]['pca_model'].inverse_transform(d_res[year_str]['pred_sample'][:, :])
        pred_values = _inv_transform(pred_values, d_info['transform']) + res

        cumsum_pred_values = np.cumsum(pred_values, axis=1)
        for i in range(d_info['n_pts']):
            ax.plot(pred_values[i, :], alpha=0.01, c='blue')  # , label=labelname
            ax2.plot(cumsum_pred_values[i, :], alpha=0.01, c='blue')

        if y is not None:
            df_epi_year = epi_year_cases_matrix(y)
            years = df_epi_year.columns if year_str == 'full' else [year_str]
            for year in years:
                ax.scatter(x=df_epi_year.index, y=df_epi_year.loc[:, year], c='red', alpha=0.2, marker='+',
                           label='true_values')
                ax2.plot(np.cumsum(df_epi_year.loc[:, year]), c='red', alpha=0.2)

        if plot_precentile:
            pred_per = np.maximum(np.percentile(pred_values, precentile, axis=0), 0)
            for i in range(len(precentile)):
                ax.plot(pred_per[i, :], c='black', alpha=0.8, linewidth=2)
                if y is not None:
                    emp_per = np.percentile(df_epi_year.T, precentile, axis=0)
                    ax.plot(emp_per[i, :], c='red', alpha=0.8, linewidth=2)
            ax.legend(handles=[true_vals, emp_precent, post_line, hbeta_precent])
        elif y is not None:
            ax.legend(handles=[true_vals, post_line])
            ax2.legend(handles=[true_line, post_line], loc='upper left')
        else:
            ax.legend(handles=[post_line])
            ax2.legend(handles=[post_line], loc='upper left')
        ax.set_title(labelname)
        ax.set_xlabel('weeks')
        ax2.set_title(labelname)
        ax2.set_xlabel('weeks')


def plot_mvn(sim=[(3, 9, 0.1, 'org', 20)]):
    fig, axes = plt.subplots(nrows=np.int(np.ceil(len(sim) / 2)), ncols=3, figsize=(15, 10))
    ind = 0
    for ax, file in zip(axes.flat, 3 * sim):
        filename = 'full_' + 'mvn' + str(file[4]) + '_' + str(file[0]) + '_' + str(file[1]) + '_' + str(file[2]) + '_' + \
                   file[3] + '.pickle'
        # labelname = 'dim='+str(file[0])+', L='+str(file[1])+', a0='+str(file[2])+', trans='+file[3]+', res='+res_method
        with (open(DIR_RES + filename, "rb")) as openfile:
            d_res, d_info = pickle.load(openfile)
        ax.hist([d_res['full']['true_vals'].iloc[:, ind], d_res['full']['pred_sample'][:, ind]], bins=20,
                label=['true', 'pred'])
        ind += 1


# load data
x_tr, y_tr = load_database(data='sj')

DIR_RES_LOO = 'C:\\Users\\itaym\\Google Drive\\University - Itay\\Msc\\Thesis\\Epi\\Results\\LOO\\pickles\\'
DIR_RES_FULL = 'C:\\Users\\itaym\\Google Drive\\University - Itay\\Msc\\Thesis\\Epi\\Results\\full\\pickles\\'
# LOO - folder
pickle_files = pathlib.Path(DIR_RES_LOO).glob("*.pickle")

list_dfs = results_table_percent(pickle_files, y_tr)
save_xls(list_dfs, DIR_RES_LOO + 'LOO_res_per.xlsx')

# res = results_table(pickle_files)
# res['MSE'] = np.mean(res.iloc[:, :18], axis=1)
# res.set_index(['dim', 'level', 'transform', 'a_0'], inplace=True)
# res.sort_index(inplace=True)


plot_week_profie(sim=[(3, 9, 1, 'square_root_0.5', 1994), (3, 9, 1, 'square_root_0.5', 2000)]
                 , res_method='emp_week', y=y_tr, plot_precentile=False)
plot_week_profie(sim=[(3, 9, 1, 'square_root_0.5', 1994), (3, 9, 1, 'square_root_0.5', 2001)]
                 , res_method='emp_week', y=y_tr, plot_precentile=False)
plot_week_profie(sim=[(3, 9, 1, 'square_root_0.5', 1994), (3, 9, 0.1, 'square_root_0.5', 1994)]
                 , res_method='emp_week', y=y_tr, plot_precentile=False)

###### save results
# res.to_excel(DIR_RES+'res.xlsx')
# plot_sample()
# plot_sample(sim=[(2, 8, 1, 'org'), (2, 8, 0.1, 'org'), (3, 6, 1, 'org'), (3, 6, 1, 'log'),
#                  (3, 6, 0.1, 'square_root')], same_scale=False)
#
# plot_sample(sim=[(2, 8, 1, 'org'), (2, 8, 0.1, 'org'), (3, 6, 1, 'org'), (3, 6, 1, 'log'),
#                  (3, 6, 0.1, 'square_root')], same_scale=True)


# full - folder
# plot_week_profie(sim=[(2, 10, 1, 'org'), (2, 10, 0.1, 'org')], y=y_tr, plot_precentile=False)
# plot_week_profie(sim=[(2, 10, 1, 'square_root'), (2, 10, 0.1, 'square_root')], y=y_tr, precentile=[50, 95],
#             res_method='normal_week')
#
# plot_week_profie(sim=[(3, 9, 1, 'org'), (3, 9, 1, 'square_root')], y=y_tr, precentile=[2.5, 97.5])
# plot_week_profie(sim=[(3, 9, 1, 'square_root'), (3, 9, 0.1, 'square_root')], y=y_tr, precentile=[25, 50, 75],
#             res_method='emp_week')
plot_week_profie(sim=[(2, 10, 1, 'square_root', 'full'), (2, 10, 0.1, 'square_root', 'full'),
                      (2, 10, 1, 'org', 'full'), (2, 10, 0.1, 'org', 'full')], y=y_tr)

# plot_week_profie(sim=[(3, 9, 1, 'org'), (3, 9, 1, 'square_root')], y=y_tr, precentile=[2.5, 97.5],
#                  res_method='normal_week')
plot_week_profie(sim=[(2, 10, 1, 'square_root', 'full'), (3, 9, 1, 'square_root', 'full')],
                 y=y_tr, plot_precentile=False)
