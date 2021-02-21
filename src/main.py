import pandas as pd
from hBeta import PThBeta
# from hBeta_fast import PThBeta
import seaborn as sns
import numpy as np
from preprocessing import plot_epidemic_year, time_stamp
from preprocessing import apply_PCA_weeks, apply_PCA_years, epi_year_cases_matrix
import matplotlib.pyplot as plt
from preprocessing import *
import collections
import argparse
import pickle
from pathlib import Path
from utils.utils_func import load_database
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import itertools

# todo:
#   check pickles
#   save time
#   run full-experiment with 3 and 4 dimensions

# TODO:
#   change that I have made:
#   -   positivity on small subset of lalonde without PCA

DIR_RES = 'C:\\Users\\itaym\\Google Drive\\University - Itay\\Msc\\Thesis\\Epi\\Results\\'


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sj', type=str, choices=['sj', 'iq', 'mvn20', 'mvn100', 'lalonde',
                                                                   'sub_lalonde', 'square_pos'])
    parser.add_argument('--dim', default=2, type=int)
    parser.add_argument('--seg_1dim', default=2, type=int)
    parser.add_argument('--n_pts', default=1000, type=int)
    parser.add_argument('--a_0', default=1, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--plot_pred', action='store_false')
    parser.add_argument('--transform', default='org', choices=['org', 'log', 'square_root'])
    parser.add_argument('--loo', action='store_true')
    parser.add_argument('--positivity', action='store_true')
    parser.add_argument('--X_and_Y', action='store_true')
    # parser.add_argument('--version', action='version', version=version)
    return parser


def loocv(y_tr, args):
    """

    Args:
        y_tr:
        args:

    Returns:

    """
    d_res = collections.defaultdict(dict)

    pt = PThBeta(seg_1dim=args.seg_1dim)
    for year in pd.unique(
            y_tr.epi_year):  # ['1991', '1994', '1998', '2000', '2005', '2007']: # [1992, 1993] pd.unique(y_tr.epi_year)
        y_one_year_out = y_tr.loc[~(y_tr['epi_year'] == year), :]
        df_pca, pca = apply_PCA_years(y_one_year_out, n_comp=args.dim, transform=args.transform, plot=False, save=False)

        pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(df_pca, n_pts=args.n_pts, gamma=args.gamma,
                                                                 a_0=args.a_0, sup_01=False)
        pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist)
        if args.plot_pred & (args.dim <= 3):
            pt.plot_sampler_grid(df_pca, pred_sample, title=year)
            file_name = args.data + '_' + str(args.dim) + '_' + str(pt.L) + '_' + str(args.a_0) + '_' + args.transform
            Path(DIR_RES + file_name).mkdir(parents=True, exist_ok=True)
            plt.savefig(DIR_RES + file_name + '\\' + str(year) + '.png')
            # plt.show()

        d_res[year] = {'true_vals': y_tr.loc[y_tr['epi_year'] == year].total_cases,
                       'pred_sample': pred_sample,
                       'pca_model': pca,
                       'pred_vals': pca.inverse_transform(np.mean(pred_sample, 0))}

    d_info = vars(args)
    d_info['level'] = pt.L

    return d_res, d_info


def full_data(y_tr, args):
    """

    Args:
        y_tr:
        args:

    Returns:

    """
    d_res = collections.defaultdict(dict)

    pt = PThBeta(seg_1dim=args.seg_1dim)
    if 'sj' in args.data:
        df_pca, pca = apply_PCA_years(y_tr, n_comp=args.dim, transform=args.transform, plot=False, save=False)
        pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(df_pca, n_pts=args.n_pts, gamma=args.gamma,
                                                                 a_0=args.a_0, sup_01=False)
    else:
        pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(y_tr, n_pts=args.n_pts, gamma=args.gamma,
                                                                 a_0=args.a_0, sup_01=False)
    pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist, n_samples=args.n_pts)
    if args.plot_pred & (args.dim <= 3):
        if 'mvn' not in args.data:
            pt.plot_sampler_grid(df_pca, pred_sample, title=args.data + '_full data')
        else:
            pt.plot_sampler_grid(y_tr, pred_sample, title=args.data + '_full data')
        file_name = 'full_' + args.data + '_' + str(args.dim) + '_' + str(pt.L) + '_' + str(args.a_0) \
                    + '_' + args.transform
        Path(DIR_RES + file_name).mkdir(parents=True, exist_ok=True)
        plt.savefig(DIR_RES + file_name + '\\' + 'full data' + '.png')
        # plt.show()

    if 'mvn' not in args.data:
        d_res['full'] = {'true_vals': y_tr.total_cases,
                         'pred_sample': pred_sample,
                         'pca_model': pca,
                         'pred_vals': pca.inverse_transform(np.mean(pred_sample, 0))}
    else:
        d_res['full'] = {'true_vals': y_tr,
                         'pred_sample': pred_sample}

    d_info = vars(args)
    d_info['level'] = pt.L
    return d_res, d_info


def positivity(X, a, args):
    d_res = collections.defaultdict(dict)

    df_treated = X.loc[a == 1, :]
    df_control = X.loc[a == 0, :]

    #############
    ## Treated ##
    #############
    pt = PThBeta(seg_1dim=args.seg_1dim)
    pt.set_int_coords(data=X, gamma=args.gamma, sup_01=False)
    pi_map_sample_treated, map_seg_pdist_treated, freq = pt.pi_hBeta_sampler(df_treated, n_pts=args.n_pts,
                                                                             gamma=args.gamma, a_0=args.a_0,
                                                                             sup_01=False, init_space=False)
    pi_treated = pt.comp_pi_post_mean(pi_map_sample_treated, map_seg_pdist_treated)
    d_res['treated'] = {'pi': pi_treated}

    #############
    ## Control ##
    #############
    pt = PThBeta(seg_1dim=args.seg_1dim)
    pt.set_int_coords(data=X, gamma=args.gamma, sup_01=False)
    pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(df_control, n_pts=args.n_pts, gamma=args.gamma,
                                                             a_0=args.a_0, sup_01=False, init_space=False)
    pi = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
    d_res['control'] = {'pi': pi}
    d_info = vars(args)
    d_info['level'] = pt.L
    return d_res, d_info


# def positivity(X, a, args):
#     d_res = collections.defaultdict(dict)
#
#     # TODO: i tried to make the dataset equal by upsampling
#     pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=args.dim))])
#     X_tr = pd.concat([X.loc[a == 1, :]]*115, ignore_index=True)
#     X_cr = pd.concat([X.loc[a == 0, :]]*1, ignore_index=True)
#     X_upsam = pd.concat([X_cr, X_tr], ignore_index=True)
#
#     # todo: At first I am trying to do it without standartization
#     # X_trans = pd.DataFrame(pipeline['scaling'].fit_transform(X_upsam), columns=X.columns)
#
#     # for col in range(X_trans.shape[1]):
#     #     mu = (1 / np.sum(weights)) * (np.sum(weights * X_trans.iloc[:, col]))
#     #     print('col:{}, mu:{:f}'.format(col, mu))
#     #     X_trans.iloc[:, col] = X_trans.iloc[:, col] - mu
#
#     principalComponents = pd.DataFrame(pipeline['pca'].fit_transform(X_upsam))
#     pca_full = pd.DataFrame(principalComponents)
#     pca_treated = pd.DataFrame(pipeline['pca'].transform(X.loc[a == 1, :]), index=X.loc[a == 1, :].index)
#     pca_control = pd.DataFrame(pipeline['pca'].transform(X.loc[a == 0, :]), index=X.loc[a == 0, :].index)
#     # pca_treated = pd.DataFrame(
#     #     pipeline['pca'].transform(pipeline['scaling'].transform(X_trans.iloc[X_cr.shape[0]:X_cr.shape[0]+X.loc[a == 1, :].shape[0], :])),
#     #     index=X.loc[a == 1, :].index)
#     # pca_control = pd.DataFrame(
#     #     pipeline['pca'].transform(pipeline['scaling'].transform(X_trans.iloc[:X_cr.shape[0], :])),
#     #     index=X.loc[a == 0, :].index)
#
#     #############
#     ## Treated ##
#     #############
#     pt = PThBeta(seg_1dim=args.seg_1dim)
#     pt.set_int_coords(data=pca_full, gamma=args.gamma, sup_01=False)
#     pi_map_sample_treated, map_seg_pdist_treated, freq = pt.pi_hBeta_sampler(pca_treated, n_pts=args.n_pts,
#                                                                              gamma=args.gamma, a_0=args.a_0,
#                                                                              sup_01=False, init_space=False)
#     # pred_sample_treated = pt.pred_map_sample(pi_map_sample_treated, map_seg_pdist_treated, n_samples=1000)
#     pi_treated = pt.comp_pi_post_mean(pi_map_sample_treated, map_seg_pdist_treated)
#     # d_res['treated'] = {'pca_model': pca, 'pi': pi_treated}
#     d_res['treated'] = {'pi': pi_treated}
#
#
#     #############
#     ## Control ##
#     #############
#
#     pt = PThBeta(seg_1dim=args.seg_1dim)
#     pt.set_int_coords(data=pca_full, gamma=args.gamma, sup_01=False)
#     pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(pca_control, n_pts=args.n_pts, gamma=args.gamma,
#                                                              a_0=args.a_0, sup_01=False, init_space=False)
#     # pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist, n_samples=1000)
#     pi = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
#
#     # d_res['control'] = {'pca_model': pca, 'pi': pi}
#     d_res['control'] = {'pi': pi}
#     d_info = vars(args)
#     d_info['level'] = pt.L
#     return d_res, d_info

# todo: here below is the pervious version of positivity -- I added a weighted version
# def positivity(X, a, args):
#     d_res = collections.defaultdict(dict)
#
#     # pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=args.dim))])
#     # scaler = StandardScaler()
#     weights = pd.Series(np.ones(a.shape), index=a.index)
#     weights.loc[a == 0] = 0.001
#     w_mat = np.array([weights, ] * X.shape[1]).T
#     # X_trans = pd.DataFrame(scaler.fit_transform(w_mat * X), columns=X.columns, index=X.index)
#     X_trans = pd.DataFrame((w_mat * X), columns=X.columns, index=X.index)
#
#     # for col in range(X_trans.shape[1]):
#     #     mu = (1 / np.sum(weights)) * (np.sum(weights * X_trans.iloc[:, col]))
#     #     print('col:{}, mu:{:f}'.format(col, mu))
#     #     X_trans.iloc[:, col] = X_trans.iloc[:, col] - mu
#
#     pca = PCA(n_components=args.dim)
#     # principalComponents = pd.DataFrame(pipeline['pca'].transform(X_trans), index=X.index)
#     # principalComponents = pd.DataFrame(pipeline['pca'].transform(X), index=X.index)
#     principalComponents = pd.DataFrame(pca.fit_transform(X_trans), index=X.index)
#     pca_full = pd.DataFrame(principalComponents)
#     pca_treated = pd.DataFrame(principalComponents.loc[a == 1, :])  # a matrix of y,x_pca
#     pca_control = pd.DataFrame(principalComponents.loc[a == 0, :])
#
#     #############
#     ## Treated ##
#     #############
#     pt = PThBeta(seg_1dim=args.seg_1dim)
#     pt.set_int_coords(data=pca_full, gamma=args.gamma, sup_01=False)
#     pi_map_sample_treated, map_seg_pdist_treated, freq = pt.pi_hBeta_sampler(pca_treated, n_pts=args.n_pts,
#                                                                              gamma=args.gamma, a_0=args.a_0,
#                                                                              sup_01=False, init_space=False)
#     # pred_sample_treated = pt.pred_map_sample(pi_map_sample_treated, map_seg_pdist_treated, n_samples=1000)
#     pi_treated = pt.comp_pi_post_mean(pi_map_sample_treated, map_seg_pdist_treated)
#     # d_res['treated'] = {'pca_model': pca, 'pi': pi_treated}
#     d_res['treated'] = {'pi': pi_treated}
#
#
#     #############
#     ## Control ##
#     #############
#
#     pt = PThBeta(seg_1dim=args.seg_1dim)
#     pt.set_int_coords(data=pca_full, gamma=args.gamma, sup_01=False)
#     pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(pca_control, n_pts=args.n_pts, gamma=args.gamma,
#                                                              a_0=args.a_0, sup_01=False, init_space=False)
#     # pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist, n_samples=1000)
#     pi = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
#
#     # d_res['control'] = {'pca_model': pca, 'pi': pi}
#     d_res['control'] = {'pi': pi}
#     d_info = vars(args)
#     d_info['level'] = pt.L
#     return d_res, d_info


def positivity_without_pca(X, a, args):
    d_res = collections.defaultdict(dict)

    dict_groups = {0: 'control', 1: 'treated'}

    for group_number, group_name in dict_groups.items():
        df = X.loc[a == group_number, :]
        pt = PThBeta(seg_1dim=args.seg_1dim)
        pt.set_int_coords(data=X, gamma=args.gamma, sup_01=False)
        pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(df, n_pts=args.n_pts, gamma=args.gamma, a_0=args.a_0,
                                                                 sup_01=False, init_space=False)
        pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist, n_samples=1000)
        pi_treated = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
        d_res[group_name] = {'X': X, 'pi': pi_treated, 'pred_sample': pred_sample}

    d_info = vars(args)
    d_info['level'] = pt.L
    return d_res, d_info


def joint_dist(X, a, y, args):
    d_res = collections.defaultdict(dict)
    dis_cols = ['black', 'hispanic', 'married', 'no_degree']

    X_con = pd.DataFrame({'age': X['age'], 'wage': (X['re75'] + X['re74']) / 2})
    X_dis = X.loc[:, dis_cols]
    df_full = pd.concat([y, X_con], axis=1)
    # df_treated = pd.concat([y.loc[a == 1], X_con.loc[a == 1, :]], axis=1)
    # df_control = pd.concat([y.loc[a == 0], X_con.loc[a == 0, :]], axis=1)

    binary_optins = list(itertools.product((0, 1), repeat=4))
    binary_optins = [(0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 0)]
    #############
    ## Treated ##
    #############
    # TODO: NOTE THAT I CHANGED HERE WHAT WILL BE SAVED --- pi ..
    pt = PThBeta(seg_1dim=args.seg_1dim)
    pt.set_int_coords(data=df_full, gamma=args.gamma, sup_01=False)
    for row in binary_optins:
        cond = (a == 1) & (X_dis['black'] == row[0]) & (X_dis['hispanic'] == row[1]) & \
               (X_dis['married'] == row[2]) & (X_dis['no_degree'] == row[3])
        num_obs = np.sum(cond)
        if num_obs > 0:
            print('num_obs: {}, row: {}'.format(num_obs, row))
            df_treated = df_full.loc[cond, :]
            pi_map_sample_treated, map_seg_pdist_treated, freq = pt.pi_hBeta_sampler(df_treated, n_pts=args.n_pts,
                                                                                     gamma=args.gamma, a_0=args.a_0,
                                                                                     sup_01=False, init_space=False)
            y_given_x_treated = pt.conditional_expected(pi_map_sample_treated, map_seg_pdist_treated)
            p_x_treated = pt.marginalizing_y(pi_map_sample_treated, map_seg_pdist_treated)

            d_res['treated' + str(row)] = {'y_given_x': y_given_x_treated, 'p_x_a': p_x_treated, 'obs': num_obs}

    # #############
    # ## Control ##
    # #############
    # pt = PThBeta(seg_1dim=args.seg_1dim)
    # pt.set_int_coords(data=df_full, gamma=args.gamma, sup_01=False)
    # for row in binary_optins:
    #     cond = (a == 0) & (X_dis['black'] == row[0]) & (X_dis['hispanic'] == row[1]) & \
    #            (X_dis['married'] == row[2]) & (X_dis['no_degree'] == row[3])
    #     num_obs = np.sum(cond)
    #     if num_obs > 0:
    #         print('**control** num_obs: {}, row: {}'.format(num_obs, row))
    #         df_control = df_full.loc[cond, :]
    #         pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(df_control, n_pts=args.n_pts,
    #                                                                  gamma=args.gamma, a_0=args.a_0,
    #                                                                  sup_01=False, init_space=False)
    #         y_given_x_control = pt.conditional_expected(pi_map_sample, map_seg_pdist)
    #         p_x_control = pt.marginalizing_y(pi_map_sample, map_seg_pdist)
    #         d_res['control' + str(row)] = {'y_given_x': y_given_x_control, 'p_x_a': p_x_control, 'obs': num_obs}

    d_info = vars(args)
    d_info['level'] = pt.L
    return d_res, d_info


# def joint_dist(X, a, y, args):
#     d_res = collections.defaultdict(dict)
#
#     pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=args.dim - 1))])
#     X_tr = pd.concat([X.loc[a == 1, :]] * 1, ignore_index=True)  # *115
#     X_cr = pd.concat([X.loc[a == 0, :]] * 1, ignore_index=True)
#     X_upsam = pd.concat([X_cr, X_tr], ignore_index=True)
#     y_upsam = pd.concat([y.loc[a == 0], pd.concat([y.loc[a == 1]] * 1, ignore_index=True)], ignore_index=True)  # *115
#
#     principalComponents = pd.DataFrame(pipeline['pca'].fit_transform(X_upsam))
#     # todo: I changed everything to be upsample
#     df_full = pd.concat([y_upsam, pd.DataFrame(principalComponents)], axis=1)
#     df_treated = pd.concat([y.loc[a == 1],
#                             pd.DataFrame(pipeline['pca'].transform(X.loc[a == 1, :]),
#                                          index=X.loc[a == 1, :].index)], axis=1)
#     df_control = pd.concat([y.loc[a == 0],
#                             pd.DataFrame(pipeline['pca'].transform(X.loc[a == 0, :]),
#                                          index=X.loc[a == 0, :].index)], axis=1)
#
#     #
#     #
#     # pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=args.dim - 1))])
#     # pipeline.fit_transform(X)
#     # # pca = PCA(n_components=args.dim - 1)
#     # # principalComponents = pd.DataFrame(pca.fit_transform(X), index=X.index)
#     # pipeline['pca'].transform(X)
#     # principalComponents = pd.DataFrame(pipeline['pca'].transform(X), index=X.index)
#     # pca_full = pd.DataFrame(principalComponents)
#     # df_full = pd.concat([y, pca_full], axis=1)
#     # pca_treated = pd.DataFrame(principalComponents.loc[a == 1, :])  # a matrix of y,x_pca
#     # pca_control = pd.DataFrame(principalComponents.loc[a == 0, :])
#
#     #############
#     ## Treated ##
#     #############
#     # TODO: NOTE THAT I CHANGED HERE WHAT WILL BE SAVED --- pi ..
#     pt = PThBeta(seg_1dim=args.seg_1dim)
#     pt.set_int_coords(data=df_full, gamma=args.gamma, sup_01=False)
#     # df_treated = pd.concat([y.loc[a == 1], pca_treated], axis=1)
#     pi_map_sample_treated, map_seg_pdist_treated, freq = pt.pi_hBeta_sampler(df_treated, n_pts=args.n_pts,
#                                                                              gamma=args.gamma, a_0=args.a_0,
#                                                                              sup_01=False, init_space=False)
#     y_given_x_treated = pt.conditional_expected(pi_map_sample_treated, map_seg_pdist_treated)
#     p_x_treated = pt.marginalizing_y(pi_map_sample_treated, map_seg_pdist_treated)
#
#     # pred_sample_treated = pt.pred_map_sample(pi_map_sample_treated, map_seg_pdist_treated, n_samples=1000)
#     # pi_treated = pt.comp_pi_post_mean(pi_map_sample_treated, map_seg_pdist_treated)
#     # d_res['treated'] = {'pca_model': pca, 'y_given_x': y_given_x_treated, 'p_x_a': p_x_treated}
#     d_res['treated'] = {'y_given_x': y_given_x_treated, 'p_x_a': p_x_treated}
#     # d_res['treated'] = {'pca_model': pca, 'pi': pi_map_sample_treated, 'seg_pdist': map_seg_pdist_treated}
#
#     #############
#     ## Control ##
#     #############
#
#     pt = PThBeta(seg_1dim=args.seg_1dim)
#     pt.set_int_coords(data=df_full, gamma=args.gamma, sup_01=False)
#     # df_control = pd.concat([y.loc[a == 0], pca_control], axis=1)
#     pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(df_control, n_pts=args.n_pts, gamma=args.gamma,
#                                                              a_0=args.a_0, sup_01=False, init_space=False)
#     y_given_x_control = pt.conditional_expected(pi_map_sample, map_seg_pdist)
#     p_x_control = pt.marginalizing_y(pi_map_sample, map_seg_pdist)
#     # pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist, n_samples=1000)
#     # pi = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
#
#     d_res['control'] = {'y_given_x': y_given_x_control, 'p_x_a': p_x_control}
#     # d_res['control'] = {'pca_model': pca, 'y_given_x': y_given_x_control, 'p_x_a': p_x_control}
#     # d_res['control'] = {'pca_model': pca, 'pi': pi_map_sample, 'seg_pdist': map_seg_pdist}
#     d_info = vars(args)
#     d_info['level'] = pt.L
#     return d_res, d_info


def joint_dist_X_and_Y(X, a, y, args):
    d_res = collections.defaultdict(dict)

    # TODO: i tried to make the dataset equal by upsampling
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=args.dim))])
    XY = pd.concat([X, y], axis=1)
    X_tr = pd.concat([XY.loc[a == 1, :]] * 115, ignore_index=True)
    X_cr = pd.concat([XY.loc[a == 0, :]] * 1, ignore_index=True)
    X_upsam = pd.concat([X_cr, X_tr], ignore_index=True)

    X_trans = pipeline['scaling'].fit_transform(X_upsam)

    principalComponents = pd.DataFrame(pipeline['pca'].fit_transform(X_trans))
    pca_full = pd.DataFrame(principalComponents)
    pca_treated = pd.DataFrame(
        pipeline['pca'].transform(X_trans[X_cr.shape[0]:X_cr.shape[0] + X.loc[a == 1, :].shape[0], :]),
        index=X.loc[a == 1, :].index)
    pca_control = pd.DataFrame(pipeline['pca'].transform(X_trans[:X_cr.shape[0], :]),
                               index=X.loc[a == 0, :].index)

    # principalComponents = pd.DataFrame(pipeline['pca'].fit_transform(X_upsam))
    # pca_full = pd.DataFrame(principalComponents)
    # pca_treated = pd.DataFrame(pipeline['pca'].transform(XY.loc[a == 1, :]), index=X.loc[a == 1, :].index)
    # pca_control = pd.DataFrame(pipeline['pca'].transform(XY.loc[a == 0, :]), index=X.loc[a == 0, :].index)

    #############
    ## Treated ##
    #############
    # TODO: NOTE THAT I CHANGED HERE WHAT WILL BE SAVED --- pi ..
    pt = PThBeta(seg_1dim=args.seg_1dim)
    pt.set_int_coords(data=pca_full, gamma=args.gamma, sup_01=False)
    pi_map_sample_treated, map_seg_pdist_treated, freq = pt.pi_hBeta_sampler(pca_treated, n_pts=args.n_pts,
                                                                             gamma=args.gamma, a_0=args.a_0,
                                                                             sup_01=False, init_space=False)
    pi_treated = pt.comp_pi_post_mean(pi_map_sample_treated, map_seg_pdist_treated)

    d_res['treated'] = {'pi': pi_treated}

    #############
    ## Control ##
    #############

    pt = PThBeta(seg_1dim=args.seg_1dim)
    pt.set_int_coords(data=pca_full, gamma=args.gamma, sup_01=False)
    pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(pca_control, n_pts=args.n_pts, gamma=args.gamma,
                                                             a_0=args.a_0, sup_01=False, init_space=False)

    pi_control = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)
    d_res['control'] = {'pi': pi_control}
    d_info = vars(args)
    d_info['level'] = pt.L
    return d_res, d_info


def main(args=None):
    """
    Main entry point for your project.
    Args:
        args : list
            A of arguments as if they were input in the command line. Leave it
            None to use sys.argv.
    """

    parser = get_parser()
    args = parser.parse_args(args)

    # TODO: I only add it for debugging
    args.data = 'lalonde'
    args.dim = 3
    args.seg_1dim = 3
    args.n_pts = 450
    args.gamma = 0.2
    print(args)

    X, y, a = load_database(data=args.data)
    # df_epi_year = epi_year_cases_matrix(y_tr)

    if args.loo:
        d_res, d_info = loocv(y, args)
        pickle_file = args.data + '_' + str(args.dim) + '_' + str(d_info['level']) + '_' + str(args.a_0) \
                      + '_' + args.transform
    elif args.data is 'sj':
        d_res, d_info = full_data(y, args)
        pickle_file = 'full_' + args.data + '_' + str(args.dim) + '_' + str(d_info['level']) + '_' + str(args.a_0) \
                      + '_' + args.transform
    elif args.data == 'square_pos':
        d_res, d_info = positivity_without_pca(X, y, args)
        pickle_file = args.data + '_' + str(args.dim) + '_' + str(d_info['level']) + '_' + str(args.a_0) \
                      + '_' + args.transform
    else:
        if (args.data == 'lalonde') | (args.data == 'sub_lalonde'):
            if args.positivity:
                print('sub lalonde positivity')
                d_res, d_info = positivity(X, a, args)
                pickle_file = 'X_' + args.data + '_' + str(args.dim) + '_' + str(d_info['level']) + '_' + str(args.a_0) \
                              + '_' + args.transform
            else:
                if args.X_and_Y:
                    d_res, d_info = joint_dist_X_and_Y(X, a, y, args)
                    pickle_file = 'XY_' + args.data + '_' + str(args.dim) + '_' + str(d_info['level']) + '_' + str(
                        args.a_0) \
                                  + '_' + 'X_and_Y'
                else:
                    d_res, d_info = joint_dist(X, a, y, args)
                    pickle_file = 'XY_' + args.data + '_' + str(args.dim) + '_' + str(d_info['level']) + '_' + str(
                        args.a_0) \
                                  + '_' + args.transform

    with open(DIR_RES + pickle_file + '.pickle', 'wb') as handle:
        pickle.dump((d_res, d_info), handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(d_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

###### to be deleted
# scaler = StandardScaler()
# X_trans = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
# weights = pd.Series(np.ones(a.shape), index=a.index)
# weights.loc[a == 0] = 0.001
#
# for col in range(X_trans.shape[1]):
#     mu = (1 / np.sum(weights)) * (np.sum(weights * X_trans.iloc[:, col]))
#     print('col:{}, mu:{:f}'.format(col, mu))
#     X_trans.iloc[:, col] = X_trans.iloc[:, col] - mu
#
# pca = PCA(n_components=3)
# pca.fit(X=X_trans)
# principalComponents = pd.DataFrame(pca.fit_transform(X_trans), index=X_trans.index)
