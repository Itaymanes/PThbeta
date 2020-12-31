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

# todo:
#   check pickles
#   save time
#   run full-experiment with 3 and 4 dimensions

DIR_RES = 'C:\\Users\\itaym\\Google Drive\\University - Itay\\Msc\\Thesis\\Epi\\Results\\'


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sj', type=str, choices=['sj', 'iq', 'mvn20', 'mvn100'])
    parser.add_argument('--dim', default=2, type=int)
    parser.add_argument('--seg_1dim', default=2, type=int)
    parser.add_argument('--n_pts', default=1000, type=int)
    parser.add_argument('--a_0', default=1, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--plot_pred', action='store_false')
    parser.add_argument('--transform', default='org', choices=['org', 'log', 'square_root'])
    parser.add_argument('--loo', action='store_true')
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
    for year in pd.unique(y_tr.epi_year): #['1991', '1994', '1998', '2000', '2005', '2007']: # [1992, 1993] pd.unique(y_tr.epi_year)
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
    if 'mvn' not in args.data:
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
    # args.data = 'mvn20'

    x_tr, y_tr = load_database(data=args.data)
    # df_epi_year = epi_year_cases_matrix(y_tr)

    args.loo = True
    if args.loo:
        d_res, d_info = loocv(y_tr, args)
        pickle_file = args.data + '_' + str(args.dim) + '_' + str(d_info['level']) + '_' + str(args.a_0) \
                      + '_' + args.transform
    else:
        d_res, d_info = full_data(y_tr, args)
        pickle_file = 'full_' + args.data + '_' + str(args.dim) + '_' + str(d_info['level']) + '_' + str(args.a_0) \
                      + '_' + args.transform
    with open(DIR_RES + pickle_file + '.pickle', 'wb') as handle:
        pickle.dump((d_res, d_info), handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(d_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
