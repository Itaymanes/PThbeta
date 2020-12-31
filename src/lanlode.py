import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from hBeta import PThBeta
import numpy as np

columns = ["training",   # Treatment assignment indicator
           "age",        # Age of participant
           "education",  # Years of education
           "black",      # Indicate whether individual is black
           "hispanic",   # Indicate whether individual is hispanic
           "married",    # Indicate whether individual is married
           "no_degree",  # Indicate if individual has no high-school diploma
           "re74",       # Real earnings in 1974, prior to study participation
           "re75",       # Real earnings in 1975, prior to study participation
           "re78"]       # Real earnings in 1978, after study end

#treated = pd.read_csv("http://www.nber.org/~rdehejia/data/nswre74_treated.txt",
#                      delim_whitespace=True, header=None, names=columns)
#control = pd.read_csv("http://www.nber.org/~rdehejia/data/nswre74_control.txt",
#                      delim_whitespace=True, header=None, names=columns)
file_names = ["http://www.nber.org/~rdehejia/data/nswre74_treated.txt",
              "http://www.nber.org/~rdehejia/data/nswre74_control.txt",
              "http://www.nber.org/~rdehejia/data/psid_controls.txt",
              "http://www.nber.org/~rdehejia/data/psid2_controls.txt",
              "http://www.nber.org/~rdehejia/data/psid3_controls.txt",
              "http://www.nber.org/~rdehejia/data/cps_controls.txt",
              "http://www.nber.org/~rdehejia/data/cps2_controls.txt",
              "http://www.nber.org/~rdehejia/data/cps3_controls.txt"]
files = [pd.read_csv(file_name, delim_whitespace=True, header=None, names=columns) for file_name in file_names]
lalonde = pd.concat(files, ignore_index=True)
lalonde = lalonde.sample(frac=1.0, random_state=42)  # Shuffle

print(lalonde.shape)
lalonde = lalonde.join((lalonde[["re74", "re75"]] == 0).astype(int), rsuffix=("=0"))
lalonde = pd.get_dummies(lalonde, columns=["education"], drop_first=True)

# Variable selection
a = lalonde.pop("training")
y = lalonde.pop("re78")
X = lalonde



pca = PCA(n_components=3)
principalComponents = pd.DataFrame(pca.fit_transform(X), index=X.index)
col = ['PC_' + str(i) for i in range(1, 3 + 1)]
pca_treated = principalComponents.loc[a == 1, :]
pca_control = principalComponents.loc[a == 0, :]

pt = PThBeta(seg_1dim=3)
pt.set_int_coords(data=principalComponents, gamma=0.2, sup_01=False)
pi_map_sample_treated, map_seg_pdist_treated, freq = pt.pi_hBeta_sampler(pca_treated, n_pts=800, gamma=0.2, a_0=1,
                                                                         sup_01=False, init_space=False)
pred_sample_treated = pt.pred_map_sample(pi_map_sample_treated, map_seg_pdist_treated, n_samples=10000)
pt.plot_sampler_grid(pca_treated, pred_sample_treated)
pi_treated = pt.comp_pi_post_mean(pi_map_sample_treated, map_seg_pdist_treated)

pt = PThBeta(seg_1dim=3)
pt.set_int_coords(data=principalComponents, gamma=0.2, sup_01=False)
pi_map_sample, map_seg_pdist, freq = pt.pi_hBeta_sampler(pca_control, n_pts=800, gamma=0.2, a_0=1,
                                                         sup_01=False, init_space=False)
pred_sample = pt.pred_map_sample(pi_map_sample, map_seg_pdist, n_samples=10000)
pt.plot_sampler_grid(pca_control, pred_sample)
pi = pt.comp_pi_post_mean(pi_map_sample, map_seg_pdist)

# plot pi
plt.subplots()
plt.plot(pi_treated, label='treated')
plt.plot(pi, label='control')
plt.legend()

plt.subplots()
plt.scatter(x=np.arange(pi_treated.shape[0]), y=pi_treated, label='treated', marker='+', s=10)
plt.scatter(x=np.arange(pi.shape[0]), y=pi, label='control', marker='.', s=10)
plt.legend()


plt.subplots()
df_pi = pd.DataFrame([pi, pi_treated]).T
plt.plot(df_pi.loc[np.sum(df_pi>0.001, axis=1) > 1, :])



start = pca.inverse_transform(pt.arr_med[df_pi.loc[np.sum(df_pi>0.0001, axis=1) > 1, :].index, :] - 0.5*pt.diff_vec)
end = pca.inverse_transform(pt.arr_med[df_pi.loc[np.sum(df_pi>0.0001, axis=1) > 1, :].index, :] + 0.5*pt.diff_vec)

c = 0

for i in range(33):
    print(np.sum(
        np.all((X.iloc[:, :8] >= start[i, :8].astype(int)) & (X.iloc[:, :8] <= end[i, :8].astype(int)), axis=1)))
    c += np.sum(
        np.all((X.iloc[:, :8] >= start[i, :8].astype(int)) & (X.iloc[:, :8] <= end[i, :8].astype(int)), axis=1))
