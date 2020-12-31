"""
Preprocessing utils code:
- load data
- Stabilize variance
- smooth data
"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy import stats


class Preprocessing:

    # def __init__(self, ):
    def preprocess(self, z, smoothing_method=None, var_stab_method=None):

        if smoothing_method is not None:
            z = self.smoothing_method(z.copy())

        if var_stab_method is not None:
            z = self.smoothing_method(z.copy())


#########
# Plots #
#########

def time_stamp(y):
    y['time_w'] = pd.Series(pd.date_range('1990-04-30', freq='W', periods=y.shape[0]))
    y.set_index('time_w', inplace=True)
    return y


def plot_epidemic_year(y):
    fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
    ax[0].plot(y['total_cases'])
    # ax[0].set_xlim([0 - 3, y.shape[0] + 3])
    ax[0].set_title('Dengue cases per week')
    ax[0].set_ylabel('total_cases')
    ax[0].get_xaxis().set_visible(False)
    sns.boxplot('epi_year', 'total_cases', data=y, ax=ax[1])
    ax[1].set_title('boxplots of Dengue cases per season')
    fig.tight_layout()


def epi_year_cases_matrix(y):
    unique_years = y['epi_year'].unique()
    # todo:
    #   to deal more properly with years with 51 or 53 weeks
    df = pd.DataFrame(data=np.zeros((51, len(unique_years))), columns=y['epi_year'].unique())
    for year in unique_years:
        df.loc[:, year] = y['total_cases'].loc[y['epi_year'] == year].reset_index(drop=True).iloc[:51]
    return df


def normalize_01(y):
    return (y-y.min())/(y.max()-y.min())


def apply_PCA_weeks(y, n_comp=2, transform='original data', plot=False):
    y_ = y.copy()
    if transform == 'log':
        y_['total_cases'] = y_['total_cases'].apply(lambda x: np.log(x + 1))
    elif transform == 'square_root':
        y_['total_cases'] = y_['total_cases'].apply(lambda x: x ** 0.5)
    z = epi_year_cases_matrix(y_)
    pca = PCA(n_components=n_comp)
    principalComponents = pca.fit_transform(z)
    col = ['PC_' + str(i) for i in range(1, n_comp + 1)]
    principaldf = pd.DataFrame(data=principalComponents
                               , columns=col)
    print('pca_explained_var\npc1: {:2f}, pc2: {:2f}'.format(pca.explained_variance_ratio_[0],
                                                             pca.explained_variance_ratio_[1]))
    if plot:
        plt.subplots()
        for pc in principaldf:
            plt.plot(principaldf[str(pc)], label=str(pc))

        plt.plot(z.mean(axis=1), c='black', label='Average across years')
        plt.title('PCA - transform: {}\nvar explained pc1:{:2.3f}, pc2:{:2.3f}'.format(transform,
                                                                                       pca.explained_variance_ratio_[0],
                                                                                       pca.explained_variance_ratio_[1]
                                                                                       ))
        plt.ylabel('Transformed dengue cases')
        plt.xlabel('Week')
        plt.legend()
        plt.tight_layout(pad=0.58)
    return principaldf


def apply_PCA_years(y, n_comp=2, transform='original data', plot=False, save=False):
    y_ = y.copy()
    if transform == 'log':
        y_['total_cases'] = y_['total_cases'].apply(lambda x: np.log(x + 1))
    elif transform == 'square_root':
        y_['total_cases'] = y_['total_cases'].apply(lambda x: x ** 0.5)
    z = epi_year_cases_matrix(y_).T
    pca = PCA(n_components=n_comp)
    principalComponents = pca.fit_transform(z)
    col = ['PC_' + str(i) for i in range(1, n_comp + 1)]
    principaldf = pd.DataFrame(data=principalComponents, columns=col, index=z.index)
    # print('pca_explained_var\npc1: {:2f}, pc2: {:2f}'.format(pca.explained_variance_ratio_[0],
    #                                                          pca.explained_variance_ratio_[1]))
    if plot:
        plt.subplots()
        for pc in principaldf:
            plt.plot(principaldf[str(pc)], label=str(pc))

        plt.plot(z.sum(axis=1)/10, c='black', label='Total cases per year\n(divided by 10)')
        plt.title('PCA - transform: {}\nvar explained pc1:{:2.3f}, pc2:{:2.3f}'.format(transform,
                                                                                       pca.explained_variance_ratio_[0],
                                                                                       pca.explained_variance_ratio_[1]
                                                                                       ))
        plt.ylabel('Transformed dengue cases')
        plt.xlabel('year')
        plt.legend()
        plt.tight_layout(pad=0.58)
    if save:
        principaldf['total_cases'] = np.sum(z, axis=1)
        principaldf['total_cases_01'] = normalize_01(principaldf['total_cases'])
        principaldf['PC_1_01'] = normalize_01(principaldf['PC_1'])
        principaldf['PC_1_Z'] = stats.zscore(principaldf['PC_1'])
        principaldf['PC_1_p'] = stats.norm.cdf(principaldf['PC_1_Z'])
        principaldf['PC_2_01'] = normalize_01(principaldf['PC_2'])
        principaldf['PC_2_Z'] = stats.zscore(principaldf['PC_2'])
        principaldf['PC_2_p'] = stats.norm.cdf(principaldf['PC_2_Z'])
        principaldf.to_csv('PC_year.csv')
    return principaldf, pca
