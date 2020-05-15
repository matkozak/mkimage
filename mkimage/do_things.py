#%%
from cell_values import *
import matplotlib.pyplot as plt
import seaborn as sns

#%%
paths = [
    '/Volumes/s-biochem-kaksonen/Mateusz/microscopy/Ede1_mutants_internal/ix81_stacks/ede1-gfp/20190108_MKY0172/processed/bgSub80px/cells/budded',
    '/Volumes/s-biochem-kaksonen/Mateusz/microscopy/Ede1_mutants_internal/ix81_stacks/ede1-gfp/20181204_MKY3682/processed/bgSub80px/cells/budded',
    '/Volumes/s-biochem-kaksonen/Mateusz/microscopy/Ede1_mutants_internal/ix81_stacks/ede1-gfp/20181206_MKY3685/processed/bgSub80px/cells/budded',
    '/Volumes/s-biochem-kaksonen/Mateusz/microscopy/Ede1_mutants_internal/ix81_stacks/ede1-gfp/20181206_MKY3688/processed/bgSub80px/cells/budded'
 ]

#%%
wt = batch_mask(paths[0], return_dict=True, save_arrays=True)
pq = batch_mask(paths[1], return_dict=True, save_arrays=True)
cc = batch_mask(paths[2], return_dict=True, save_arrays=True)
pqcc = batch_mask(paths[3], return_dict=True, save_arrays=True)

#%%
wt_all = np.concatenate(list(wt.values()))
pq_all = np.concatenate(list(pq.values()))
cc_all = np.concatenate(list(cc.values()))
pqcc_all = np.concatenate(list(pqcc.values()))

wt_r

#%%
all_pixels = dict(wt_all=wt_all, pq_all=pq_all, cc_all=cc_all, pqcc_all=pqcc_all)
all_pixels_rescaled = dict()
for key, value in all_pixels_rescaled.items():
    all_pixels_rescaled[key] = rescale_to_float(value)

#%%
for key, value in all_pixels.items():
    sns.distplot(value, label=key, kde=False)
plt.legend()
plt.show()

for key, value in all_pixels_rescaled.items():
    sns.distplot(value, label=key)
plt.legend()
plt.show()

#%%
for key, value in all_pixels.items():
    #sns.distplot(value, label = key, hist_kws=dict({'cumulative':True}), kde_kws=dict({'cumulative':True}))
    sns.kdeplot(value, label = key, cumulative=True)
plt.legend()
plt.show()

#%%
import scipy.integrate
import statsmodels.distributions.empirical_distribution as ed

def my_cdf(a, rescale=False):
    a = np.sort(a)
    if rescale:
        a = rescale_to_float(a)
    n = len(a)
    cdf = np.linspace(1, n, n)/n
    return np.column_stack((a,cdf))

def my_cdf_trapz(a, rescale=False):
    a = my_cdf(a, rescale = rescale)
    integral = scipy.integrate.trapz(a[:,1], a[:,0])
    return integral

def ecdf_quad(a, rescale=False):
    if rescale:
        a = rescale_to_float(a)
    ecdf = ed.ECDF(a)
    integral = scipy.integrate.quad(ecdf, min(a), max(a))
    return integral

#%%
print(my_cdf_trapz(wt_all, rescale=True))
print(ecdf_quad(wt_all, rescale=True))

#%%
import timeit
import warnings
warnings.filterwarnings('ignore')
print(timeit.timeit('my_cdf_trapz(wt_all)', globals=globals(), number=1000))
print(timeit.timeit('ecdf_quad(wt_all)', globals=globals(), number = 1000))

#%%
wt_int_r = {}
for key, value in wt.items():
    wt_int_r[key] = my_cdf_trapz(value, rescale=True)

#%%
sns.distplot(list(wt_int.values()), bins=10)
plt.show()
sns.distplot(list(wt_int_r.values()), bins=10)
plt.show()

#%%
import pandas as pd
ph = batch_mask('/Users/kozak/Documents/unige/scripts/python/test_data/distribution/ph', pattern='*.tif', return_dict=True)
print(ph)
ph_pd = {}
for key, value in ph.items():
    ph_pd[key] = np.array([np.mean(value), my_cdf_trapz(value, rescale=True)])

ph_pd = pd.DataFrame.from_dict(ph_pd, orient='index', columns=['mean', 'clust'])

sns.relplot(data=ph_pd, x='mean', y='clust')
