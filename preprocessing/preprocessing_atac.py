import numpy as np
import pandas as pd
from time import time
import scanpy as sc
from scipy.io import mmread
import random
import os
import gc
import sys
import argparse
import importlib.util
import inspect
from sklearn.model_selection import train_test_split

try:
    import torch
except Exception:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

# ------------------------------------------------------------
# scMAGCA Preprocessing Compatibility Layers
# ------------------------------------------------------------
def read_dataset(adata, transpose=False, test_split=False, copy=False):
    if isinstance(adata, sc.AnnData):
        if copy: adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    if transpose:
        adata = adata.transpose()
    adata.obs["DCA_split"] = "train"
    return adata

def preprocess_dataset(adata, size_factors=False, normalize_input=True, logtrans_input=True):
    import scipy as _sp
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x)) if len(x) > 0 else 1.0
        return np.log1p(x / exp)
    
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata)
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.raw.X.A if _sp.sparse.issparse(adata.raw.X) else adata.raw.X)
    )
    if logtrans_input: sc.pp.log1p(adata)
    if normalize_input: sc.pp.scale(adata)
    return adata

def geneSelection(data, n=None, threshold=0, atleast=10, decay=1.5, xoffset=5, yoffset=.02):
    from scipy import sparse as _sparse
    if _sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (1 - zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:, detected] > threshold
        logs = np.zeros_like(data[:, detected]) * np.nan
        logs[mask] = np.log2(data[:, detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    if n is not None:
        up, low = 10, 0
        for _ in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n: break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
    return selected

def GetCluster(X, res, n):
    adata0 = sc.AnnData(X)
    sc.pp.neighbors(adata0, n_neighbors=n, use_rep="X")
    sc.tl.louvain(adata0, resolution=res)
    y_pred = np.asarray(adata0.obs["louvain"], dtype=int)
    return np.unique(y_pred).shape[0]

def _align_modalities(adata_atac, adata_rna):
    common = pd.Index(adata_atac.obs_names).intersection(adata_rna.obs_names)
    if len(common) == 0:
        raise ValueError("No shared barcodes found between modalities.")
    print(f"Alignment: {len(common)} shared barcodes found.")
    return adata_atac[common].copy(), adata_rna[common].copy()

# ------------------------------------------------------------
# scMAGCL Module Integration
# ------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
# Explicitly targeting the scMAGCL-main folder as specified
_SCMAGCL_ROOT = os.path.join(_PARENT_DIR, "scMAGCL-main")

def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

gcl_cfg, CellDataset, device, gcl_train, gcl_test = [None] * 5

def _ensure_scMAGCL_loaded():
    global gcl_cfg, CellDataset, device, gcl_train, gcl_test
    if gcl_cfg is not None: return
    
    gcl_utils = _load_local_module("utils", os.path.join(_SCMAGCL_ROOT, "utils.py"))
    _load_local_module("scMAGCL", os.path.join(_SCMAGCL_ROOT, "scMAGCL.py"))
    gcl_config = _load_local_module("config", os.path.join(_SCMAGCL_ROOT, "config.py"))
    gcl_main = _load_local_module("main", os.path.join(_SCMAGCL_ROOT, "main.py"))
    
    gcl_cfg, CellDataset, device, gcl_train, gcl_test = gcl_config.config, gcl_utils.CellDataset, gcl_utils.device, gcl_main.train, gcl_main.test

def _read_modality(name, h5ad_path=None, tenx_dir=None):
    if h5ad_path:
        adata = sc.read_h5ad(h5ad_path)
        print(f"Loaded {name} (h5ad): {adata.shape}")
        return adata
    if tenx_dir:
        adata = sc.read_10x_mtx(tenx_dir, var_names="gene_symbols", make_unique=True)
        print(f"Loaded {name} (10x): {adata.shape}")
        return adata
    raise ValueError(f"Input path for {name} missing.")

def main_atac_rna():
    parser = argparse.ArgumentParser(description="scMAGCL ATAC+RNA Pipeline")
    # Restore all missing arguments
    parser.add_argument("--atac_h5ad", default=None)
    parser.add_argument("--rna_h5ad", default=None)
    parser.add_argument("--atac_10x_dir", default=None)
    parser.add_argument("--rna_10x_dir", default=None)
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--filter1", action="store_true", default=False)
    parser.add_argument("--filter2", action="store_true", default=False)
    parser.add_argument("--f1", default=2000, type=int)
    parser.add_argument("--f2", default=2000, type=int)
    parser.add_argument("--no_clr", action="store_true", default=False)
    parser.add_argument("--no_scale", action="store_true", default=False)
    parser.add_argument("--n_clusters", type=int, default=9)
    parser.add_argument("--resolution", type=float, default=0.08)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--print_k_only", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_cl", type=float, default=None)
    parser.add_argument("--lambda_byol", type=float, default=None)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--sweep", action="store_true", default=False)
    parser.add_argument("--sweep_param", type=str, default=None)
    parser.add_argument("--sweep_values", type=str, default=None)
    
    args = parser.parse_args()

    _ensure_scMAGCL_loaded()
    
    # Defaults from config
    args.seed = int(gcl_cfg["seed"]) if args.seed is None else int(args.seed)
    args.epochs = int(gcl_cfg["epochs"]) if args.epochs is None else int(args.epochs)
    args.lr = float(gcl_cfg["lr"]) if args.lr is None else float(args.lr)
    args.lambda_cl = float(gcl_cfg["lambda_cl"]) if args.lambda_cl is None else float(args.lambda_cl)
    args.lambda_byol = float(gcl_cfg.get("lambda_byol", 1.0)) if args.lambda_byol is None else float(args.lambda_byol)

    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

    adata_atac = _read_modality("ATAC", h5ad_path=args.atac_h5ad, tenx_dir=args.atac_10x_dir)
    adata_rna = _read_modality("RNA", h5ad_path=args.rna_h5ad, tenx_dir=args.rna_10x_dir)
    adata_atac, adata_rna = _align_modalities(adata_atac, adata_rna)

    x1 = adata_atac.X.toarray() if hasattr(adata_atac.X, "toarray") else adata_atac.X
    x2 = adata_rna.X.toarray() if hasattr(adata_rna.X, "toarray") else adata_rna.X

    if args.filter1:
        x1 = x1[:, geneSelection(x1, n=args.f1)]
    if args.filter2:
        x2 = x2[:, geneSelection(x2, n=args.f2)]

    def _run_prep(x, no_clr, no_scale):
        adata = sc.AnnData(x)
        if no_clr or no_scale:
            adata.raw = adata.copy()
            sc.pp.normalize_total(adata)
            if not no_clr:
                def clr_row(r):
                    s = np.sum(np.log1p(r[r > 0]))
                    ev = np.exp(s / len(r)) if len(r) > 0 else 1.0
                    return np.log1p(r / (ev if ev != 0 else 1.0))
                adata.X = np.apply_along_axis(clr_row, 1, adata.raw.X)
            sc.pp.log1p(adata)
            if not no_scale: sc.pp.scale(adata)
            return adata.X
        else:
            adata = preprocess_dataset(read_dataset(adata))
            return adata.X.A if hasattr(adata.X, "A") else adata.X

    x1_proc = _run_prep(x1, args.no_clr, args.no_scale)
    x2_proc = _run_prep(x2, args.no_clr, args.no_scale)
    X = np.concatenate([x1_proc, x2_proc], axis=1).astype('float32')
    X = np.nan_to_num(X)

    y = np.zeros(X.shape[0], dtype=np.int64)
    if args.label_csv:
        label_df = pd.read_csv(args.label_csv)
        y, _ = pd.factorize(label_df.iloc[:, 1].values, sort=True)

    if args.n_clusters <= 0:
        args.n_clusters = GetCluster(X, res=args.resolution, n=args.knn_k)

    if args.print_k_only:
        print(f"n_clusters={args.n_clusters}")
        sys.exit(0)

    os.makedirs(args.save_dir, exist_ok=True)

    sweep_list = [float(v) for v in args.sweep_values.split(",")] if args.sweep and args.sweep_values else [None]

    for sweep_val in sweep_list:
        cur_lcl = sweep_val if args.sweep_param == "lambda_cl" else args.lambda_cl
        cur_lby = sweep_val if args.sweep_param == "lambda_byol" else args.lambda_byol
        
        for i in range(args.n_runs):
            run_seed = args.seed + i
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run_seed)
            train_loader = torch.utils.data.DataLoader(CellDataset(X_train, y_train), batch_size=128, shuffle=True)
            test_loader = torch.utils.data.DataLoader(CellDataset(X_test, y_test), batch_size=128, shuffle=False)

            # train returns 4 values: best_epoch, min_loss, best_z_test, best_y_test
            best_epoch, min_loss, z_test, y_true_test = gcl_train(
                train_loader=train_loader, test_loader=test_loader, input_dim=X.shape[1],
                graph_head=gcl_cfg['graph_head'], phi=gcl_cfg['phi'], gcn_dim=gcl_cfg['gcn_dim'],
                mlp_dim=gcl_cfg['mlp_dim'], prob_feature=gcl_cfg['prob_feature'],
                prob_edge=gcl_cfg['prob_edge'], tau=gcl_cfg['tau'], alpha=gcl_cfg['alpha'],
                beta=gcl_cfg['beta'], lambda_cl=cur_lcl, dropout=gcl_cfg['dropout'],
                lr=args.lr, seed=run_seed, epochs=args.epochs, device=device,
                phi1=gcl_cfg['phi1'], lambda_byol=cur_lby
            )
            
            results = gcl_test(z_test, y_true_test, args.n_clusters, run_seed)
            print(f"Run {i+1} Result: CA={results['CA']:.4f}, NMI={results['NMI']:.4f}, ARI={results['ARI']:.4f}")
            
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main_atac_rna()