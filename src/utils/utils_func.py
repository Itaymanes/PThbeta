import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pandas import ExcelWriter
DIR = 'C:\\Users\\itaym\\Google Drive\\University - Itay\\Msc\\Thesis\\Epi\\Data\\Competition\\Stations_org\\'



def load_database(data='sj'):
    """
    load
    Args:
        data (str): the data that will be loaded is either sj (san-juan) or iq (iqutious)
    Returns:
        x_tr (pd.DataFrame), y_tr (pd.DataFrame)
    """
    np.random.seed(0)
    mn = [0, -1, 1]
    sgma = [[1, 0.8, 0.5], [0.8, 1, 0.7], [0.5, 0.7, 1]]  # diagonal covariance
    x_tr = 0
    if data == 'sj':
        x_tr = pd.read_csv(DIR + 'x_tr_sj.csv')
        y_tr = pd.read_csv(DIR + 'y_tr_sj.csv')
    elif data == 'iq':
        x_tr = pd.read_csv(DIR + 'x_tr_iq.csv')
        y_tr = pd.read_csv(DIR + 'y_tr_iq.csv')
    elif data == 'mvn20':
        y_tr = pd.DataFrame(np.random.multivariate_normal(mn, sgma, 20))
    elif data == 'mvn100':
        y_tr = pd.DataFrame(np.random.multivariate_normal(mn, sgma, 100))
    print("data of {} will be loaded".format(data))
    return x_tr, y_tr


def _transform(y, trans='org'):
    if trans == 'square_root':
        return y ** 0.5
    elif trans == 'log':
        return np.log(y + 1)
    else:
        return y


def _inv_transform(y, trans='org'):
    if trans == 'square_root':
        return y ** 2
    elif trans == 'log':
        return np.exp(y) - 1
    else:
        return y


def sample_pca_residuals_distribution(df_epi_year, n_comp, trans, n_pts, res_method):
    """
    Compute the distribution of residuals and sample (n_pts x features) out of it
    Args:
        df_epi_year (pd.DataFrame | np.array):
        n_comp (int) :
        trans (str):
        n_pts (int):
        res_method (str):

    Returns:

    """
    z = _transform(df_epi_year, trans=trans)
    pca = PCA(n_components=n_comp)
    principalComponents = pca.fit_transform(z)
    y_pca = _inv_transform(pca.inverse_transform(principalComponents), trans=trans)
    if res_method is "None":
        return np.zeros(shape=(n_pts, z.shape[1]))
    elif res_method == 'emp_full':
        ###### (1) residuals as one long vector ####
        res = np.reshape(y_pca - np.array(df_epi_year), -1)
        return np.random.choice(res, size=(n_pts, z.shape[1]))
    elif res_method == 'emp_week':
        ###### (2) residuals sampled differently per week ####
        res = y_pca - np.array(df_epi_year)
        a = [np.random.choice(res[:, _], size=n_pts, replace=True) for _ in range(z.shape[1])]
        return np.array(a).T
    elif res_method == 'normal_week':
        res = y_pca - np.array(df_epi_year)
        mu, sigma = np.mean(res, axis=0), np.std(res, axis=0, ddof=1)
        return np.random.normal(mu, sigma, size=(n_pts, z.shape[1]))
    else:
        raise TypeError('a method of residuals is needed in order to compute the posterior predictive distribution')


def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, str(1990+n))
        writer.save()