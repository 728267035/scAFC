import numpy as np
import pandas as pd
import scanpy as sc
import h5py
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')

results_file = 'D:/scAFC/scAFC-master/scdata/h5/Bladder.h5'
data_mat = h5py.File(results_file, "r+")
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])

adata = sc.AnnData(x)
adata.var_names_make_unique()
print(adata)
sc.pl.highest_expr_genes(adata, n_top=20, )
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
print(adata.X)
adata.write_h5ad(r'D:/scAFC/scAFC-master/scdata/Bladder.h5ad')
print(adata.obs.columns)
X = adata.X

np.savetxt('D:/scAFC/scAFC-master/data/Bladder.txt', X, fmt='%f')
np.savetxt('D:/scAFC/scAFC-master/data/Bladder.csv', X, delimiter=',', fmt='%f')
