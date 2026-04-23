import os
import importlib
import torch
import random
import numpy as np
import h5py
import scanpy as sc
from scipy.sparse import issparse, csr_matrix
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.preprocessing import LabelEncoder

# Global device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class CellDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for single-cell data handling both dense and sparse matrices.
    """
    def __init__(self, X, y):
        if issparse(X):
            self.X = X.tocsr()
        else:
            self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if issparse(self.X):
            x = self.X[idx].toarray().squeeze()
        else:
            x = self.X[idx]
        x = torch.tensor(x, dtype=torch.float32)
        return x, self.y[idx]

def qc_filter(adata, min_genes=200, max_genes=5000, min_cells=3):
    """
    Performs quality control filtering on cells and genes.
    """
    n_cells_orig, n_genes_orig = adata.shape
    print(f"Pre-filtering: {n_cells_orig} cells x {n_genes_orig} genes")

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    n_cells, n_genes = adata.shape
    print(f"Post-filtering: {n_cells} cells x {n_genes} genes")
    print(f"Retention: Cells {n_cells/n_cells_orig:.1%}, Genes {n_genes/n_genes_orig:.1%}")

    return adata

def normalize(adata, target_sum=1e4):
    """
    Applies total count normalization and log1p transformation.
    """
    print(f"Normalizing data (target_sum={target_sum:,.0f})...")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Clean non-finite values
    if not np.all(np.isfinite(adata.X.data if issparse(adata.X) else adata.X)):
        print("Warning: Non-finite values detected. Replacing with zeros.")
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

    return adata

def select_highly_variable_genes(adata, n_top_genes=2000):
    """
    Selects highly variable genes (HVGs) using the Seurat flavor or variance-based fallback.
    """
    print(f"Selecting top {n_top_genes} highly variable genes...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    except Exception as e:
        print(f"Seurat method failed: {e}. Falling back to variance-based selection.")
        X_dense = adata.X.toarray() if issparse(adata.X) else adata.X
        gene_var = np.nan_to_num(np.var(X_dense, axis=0))
        top_indices = np.argsort(gene_var)[-n_top_genes:]
        adata.var['highly_variable'] = False
        adata.var.iloc[top_indices, adata.var.columns.get_loc('highly_variable')] = True
    
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"Final feature dimensions: {adata.shape[1]} genes")
    return adata

def loader_construction(data_path):
    """
    Constructs DataLoaders from HDF5 files supporting multiple data formats.
    """
    print(f"Loading data from: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # Format 0: Simple mode (X, y) - Preprocessed features
        if 'X' in f and 'y' in f and 'obs' not in f:
            print("Detected simple format (X, y).")
            X_all = f['X'][()]
            y_all = f['y'][()].squeeze()
            
            if isinstance(y_all[0], bytes):
                y_all = np.array([v.decode('utf-8') for v in y_all])

            le = LabelEncoder()
            y_encoded = le.fit_transform(y_all)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            train_loader = DataLoader(CellDataset(X_train, y_train), batch_size=128, shuffle=True)
            test_loader = DataLoader(CellDataset(X_test, y_test), batch_size=128, shuffle=False)
            
            return train_loader, test_loader, X_all.shape[1]

        # Format 1: Standard (X, y, cell_barcodes)
        if 'X' in f and 'y' in f and 'cell_barcodes' in f:
            X_all = csr_matrix(f['X'][()]) if not issparse(f['X'][()]) else f['X'][()]
            y_all = f['y'][()].astype(str)
        
        # Format 2: Group/Baron format (X, obs)
        elif 'X' in f and 'obs' in f:
            X_all = csr_matrix(f['X'][()])
            y_all = None
            # Search for typical label keys in obs
            possible_keys = ['cell_type', 'celltype', 'Group', 'cluster', 'labels']
            for key in possible_keys:
                if key in f['obs']:
                    y_all = f[f'obs/{key}'][()].astype(str)
                    break
            if y_all is None:
                print("Warning: No labels found. Generating dummy labels.")
                y_all = np.array([f'Cluster_{i % 10}' for i in range(X_all.shape[0])])
                
        # Format 3: Sparse matrix format (exprs)
        elif 'exprs' in f:
            data, indices, indptr = f['exprs/data'][()], f['exprs/indices'][()], f['exprs/indptr'][()]
            shape = tuple(f['exprs/shape'][()])
            X_all = csr_matrix((data, indices, indptr), shape=shape)
            y_all = f['obs/cell_type1'][()].astype(str)
            
        else:
            raise ValueError(f"Unsupported data format. Keys: {list(f.keys())}")

    # Standard preprocessing pipeline for raw counts
    adata = sc.AnnData(X=X_all, obs={'cell_type': y_all})
    adata = normalize(adata)
    adata = select_highly_variable_genes(adata)

    le = LabelEncoder()
    y_encoded = le.fit_transform(adata.obs['cell_type'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        adata.X, y_encoded, test_size=0.2, random_state=1
    )

    def sparse_collate(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y

    train_loader = DataLoader(CellDataset(X_train, y_train), batch_size=128, shuffle=True, collate_fn=sparse_collate)
    test_loader = DataLoader(CellDataset(X_test, y_test), batch_size=128, shuffle=False, collate_fn=sparse_collate)

    return train_loader, test_loader, adata.shape[1]

def setup_seed(seed):
    """
    Ensures reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cluster_acc(y_true, y_pred):
    """
    Computes Clustering Accuracy (ACC) using the Hungarian algorithm.
    """
    if isinstance(y_true[0], (bytes, str)):
        y_true = LabelEncoder().fit_transform(y_true)

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) / y_pred.size

def evaluate(y_true, y_pred, adata=None, method=None):
    """
    Evaluates clustering performance using ACC, NMI, ARI, and optionally scIB metrics.
    """
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)

    clisi, casw = None, None
    try:
        # Optional scIB metrics
        if adata is not None and method is not None:
            scib = importlib.import_module("scib")
            clisi = scib.metrics.clisi_graph(adata, label_key="cell_type", type_="embed", use_rep=method)
            casw = scib.metrics.silhouette(adata, label_key="cell_type", embed=method)
    except Exception as e:
        print(f"Warning: scib metrics calculation failed: {e}")

    return acc, 0, nmi, ari, homo, comp, clisi, casw