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
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder

# Set device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CellDataset(torch.utils.data.Dataset):
    """Dataset class for single-cell feature matrices."""
    def __init__(self, X, y):
        self.X = X.tocsr() if issparse(X) else X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].toarray().squeeze() if issparse(self.X) else self.X[idx]
        return torch.tensor(x, dtype=torch.float32), self.y[idx]

def qc_filter(adata, min_genes=200, max_genes=5000, min_cells=3):
    """Perform standard quality control filtering on AnnData."""
    print(f"Pre-QC shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"Post-QC shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    return adata

def normalize(adata, target_sum=1e4):
    """Log-normalization and numerical stability cleanup."""
    print(f"Normalization: Target sum = {target_sum}")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Cleanup non-finite values
    X_data = adata.X if not issparse(adata.X) else adata.X.data
    nan_mask = ~np.isfinite(X_data)
    if np.any(nan_mask):
        print(f"Warning: Detected {np.sum(nan_mask)} non-finite values. Replacing with 0.")
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    return adata

def select_highly_variable_genes(adata, n_top_genes=2000):
    """Select highly variable genes with fallback to variance-based selection."""
    print(f"Feature selection: Extracting top {n_top_genes} HVGs")
    
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    except (ValueError, RuntimeWarning) as e:
        print(f"Seurat HVG selection failed: {str(e)[:50]}. Falling back to variance ranking.")
        X_dense = adata.X.toarray() if issparse(adata.X) else adata.X
        gene_var = np.nan_to_num(np.var(X_dense, axis=0), nan=0.0)
        hvg_indices = np.argsort(gene_var)[-n_top_genes:]
        adata.var['highly_variable'] = False
        adata.var.iloc[hvg_indices, adata.var.columns.get_loc('highly_variable')] = True
    
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"Final feature dimension: {adata.shape[1]}")
    return adata

def loader_construction(data_path):
    """Main pipeline for data loading and DataLoader construction."""
    print(f"Loading dataset from: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # 动态检测标签名称：兼容小写的y，大写的Y，或者是labels
        label_key = None
        for k in ['y', 'Y', 'labels']:
            if k in f:
                label_key = k
                break

        # Format 0: Preprocessed features (X, y/Y/labels)
        if 'X' in f and label_key is not None and 'obs' not in f:
            print(f"Detected preprocessed feature format (using label key: '{label_key}').")
            X_all = f['X'][()]
            y_all = f[label_key][()].squeeze()
            if isinstance(y_all[0], bytes):
                y_all = np.array([v.decode('utf-8') for v in y_all])

            le = LabelEncoder()
            y_encoded = le.fit_transform(y_all)
            n_classes = len(le.classes_)

            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            train_loader = DataLoader(CellDataset(X_train, y_train), batch_size=128, shuffle=True)
            test_loader = DataLoader(CellDataset(X_test, y_test), batch_size=128, shuffle=False)
            
            print(f"Load successful: {X_all.shape[0]} cells, {X_all.shape[1]} features, {n_classes} classes.")
            return train_loader, test_loader, X_all.shape[1]

        # Format 1/2: AnnData-like or Baron format (X, obs)
        elif 'X' in f and ('obs' in f or 'cell_barcodes' in f):
            print("Detected AnnData/HDF5 structure.")
            X_all = f['X'][()]
            
            # Label detection logic
            y_all = None
            if 'obs' in f:
                possible_keys = ['cell_type', 'celltype', 'cluster', 'labels', 'Group', 'annotation']
                for key in possible_keys:
                    if key in f['obs']:
                        y_all = f[f'obs/{key}'][()].astype(str)
                        break
            elif label_key is not None:
                y_all = f[label_key][()].astype(str)

            if y_all is None:
                print("Warning: No labels found. Generating dummy clusters.")
                y_all = np.array([f"Cluster_{i % 10}" for i in range(X_all.shape[0])])

            adata = sc.AnnData(X=X_all, obs={'cell_type': y_all})
            adata = normalize(adata)
            adata = select_highly_variable_genes(adata)

            le = LabelEncoder()
            y_encoded = le.fit_transform(adata.obs['cell_type'])
            
            X_train, X_test, y_train, y_test = train_test_split(
                adata.X, y_encoded, test_size=0.2, random_state=1
            )

            train_loader = DataLoader(CellDataset(X_train, y_train), batch_size=128, shuffle=True)
            test_loader = DataLoader(CellDataset(X_test, y_test), batch_size=128, shuffle=False)
            
            return train_loader, test_loader, adata.shape[1]

        else:
            # 稍微加了一个报错提示，如果再遇到不支持的格式，会打印出文件里到底有什么键
            keys = list(f.keys())
            raise ValueError(f"Unsupported file format at {data_path}. Available keys: {keys}")

def setup_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cluster_acc(y_true, y_pred):
    """Calculate clustering accuracy via Hungarian algorithm alignment."""
    if isinstance(y_true[0], (bytes, str)):
        y_true = LabelEncoder().fit_transform(y_true)
    
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size

def evaluate(y_true, y_pred, adata=None, method=None):
    """Comprehensive evaluation of clustering metrics."""
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)

    clisi, casw = None, None
    try:
        if adata is not None and method is not None:
            import scib
            clisi = scib.metrics.clisi_graph(adata, label_key="cell_type", type_="embed", use_rep=method)
            casw = scib.metrics.silhouette(adata, label_key="cell_type", embed=method)
    except Exception as e:
        print(f"Integration metrics (scib) skipped: {e}")

    return acc, 0, nmi, ari, homo, comp, clisi, casw