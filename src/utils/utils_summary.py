import pickle
import numpy as np
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

def _dyadic_cube_to_samples222(X, pt):
    min_dyc = pt.arr_med - pt.diff_vec / 2
    max_dyc = pt.arr_med + pt.diff_vec / 2
    int_mat = pt.arr_ind @ pt.weight_vec
    d = dict.fromkeys(int_mat, 0)
    if X.shape[1] == 2:
        for pt_cell in d:
            temp = ((X.iloc[:, 0] < max_dyc[pt_cell, 0]) & (X.iloc[:, 1] < max_dyc[pt_cell, 1])) \
                   & ((X.iloc[:, 0] > min_dyc[pt_cell, 0]) & (X.iloc[:, 1] > min_dyc[pt_cell, 1]))
            d[pt_cell] = temp.loc[temp == True].index
    else:
        for pt_cell in d:
            temp = np.where(((X.iloc[:, 0] < max_dyc[pt_cell, 0]) & (X.iloc[:, 1] < max_dyc[pt_cell, 1]) &
                                   (X.iloc[:, 2] < max_dyc[pt_cell, 2])) &
                                  ((X.iloc[:, 0] > min_dyc[pt_cell, 0]) & (X.iloc[:, 1] > min_dyc[pt_cell, 1]) &
                                   (X.iloc[:, 2] > min_dyc[pt_cell, 2])))[0]
            d[pt_cell] = temp.loc[temp == True].index
    return d
