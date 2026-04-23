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
    import torch  # optional at import-time; required for training
except Exception:
    torch = None

# 导入psutil用于内存监控
try:
    import psutil
except ImportError:
    psutil = None

# ------------------------------------------------------------
# ATAC+RNA 预处理依赖（历史来自 scMAGCA；当前仓库不再包含该包）
# 为保持原脚本的处理顺序与结果，本文件提供“同名函数”的兼容实现：
# - read_dataset
# - preprocess_dataset
# - geneSelection
# - GetCluster
# 若外部环境已安装 scMAGCA，则优先使用其实现。
# ------------------------------------------------------------
try:
    from scMAGCA.preprocess import read_dataset as read_dataset  # type: ignore
    from scMAGCA.preprocess import preprocess_dataset as preprocess_dataset  # type: ignore
    from scMAGCA.utils import geneSelection as geneSelection  # type: ignore
    from scMAGCA.utils import GetCluster as GetCluster  # type: ignore
except Exception:
    def read_dataset(adata, transpose=False, test_split=False, copy=False):
        import scipy as _sp
        from sklearn.model_selection import train_test_split as _train_test_split

        if isinstance(adata, sc.AnnData):
            if copy:
                adata = adata.copy()
        elif isinstance(adata, str):
            adata = sc.read(adata)
        else:
            raise NotImplementedError

        norm_error = "Make sure that the dataset (adata.X) contains unnormalized count data."
        assert "n_count" not in adata.obs, norm_error

        if adata.X.size < 50e6:
            if _sp.sparse.issparse(adata.X):
                assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
            else:
                assert np.all(adata.X.astype(int) == adata.X), norm_error

        if transpose:
            adata = adata.transpose()

        if test_split:
            _, test_idx = _train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
            spl = pd.Series(["train"] * adata.n_obs)
            spl.iloc[test_idx] = "test"
            adata.obs["DCA_split"] = spl.values
        else:
            adata.obs["DCA_split"] = "train"

        adata.obs["DCA_split"] = adata.obs["DCA_split"].astype("category")
        print("### Autoencoder: Successfully preprocessed {} features and {} cells.".format(adata.n_vars, adata.n_obs))
        return adata

    def preprocess_dataset(adata, size_factors=False, normalize_input=True, logtrans_input=True):
        import scipy as _sp

        def seurat_clr(x):
            s = np.sum(np.log1p(x[x > 0]))
            exp = np.exp(s / len(x))
            return np.log1p(x / exp)

        adata.raw = adata.copy()
        sc.pp.normalize_total(adata)

        if size_factors:
            adata.obs["size_factors"] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        else:
            adata.obs["size_factors"] = 1.0

        adata.X = np.apply_along_axis(
            seurat_clr, 1, (adata.raw.X.A if _sp.sparse.issparse(adata.raw.X) else adata.raw.X)
        )

        if logtrans_input:
            sc.pp.log1p(adata)
        if normalize_input:
            sc.pp.scale(adata)
        return adata

    def GetCluster(X, res, n):
        adata0 = sc.AnnData(X)
        if adata0.shape[0] > 200000:
            np.random.seed(adata0.shape[0])
            adata0 = adata0[np.random.choice(adata0.shape[0], 200000, replace=False)]
        sc.pp.neighbors(adata0, n_neighbors=n, use_rep="X")
        sc.tl.louvain(adata0, resolution=res)
        y_pred_init = adata0.obs["louvain"]
        y_pred_init = np.asarray(y_pred_init, dtype=int)
        if np.unique(y_pred_init).shape[0] <= 1:
            raise RuntimeError(
                "Error: There is only a cluster detected. The resolution:"
                + str(res)
                + " is too small, please choose a larger resolution!!"
            )
        print("Estimated n_clusters is: ", np.shape(np.unique(y_pred_init))[0])
        return np.shape(np.unique(y_pred_init))[0]

    def geneSelection(data, threshold=0, atleast=10, yoffset=.02, xoffset=5, decay=1.5, n=None,
                      plot=True, markers=None, genes=None, figsize=(6, 3.5),
                      markeroffsets=None, labelsize=10, alpha=1, verbose=1):
        from scipy import sparse as _sparse
        import matplotlib.pyplot as plt
        import seaborn as sns

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

        lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
        zeroRate[lowDetection] = np.nan
        meanExpr[lowDetection] = np.nan

        if n is not None:
            up = 10
            low = 0
            for _ in range(100):
                nonan = ~np.isnan(zeroRate)
                selected = np.zeros_like(zeroRate).astype(bool)
                selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
                if np.sum(selected) == n:
                    break
                elif np.sum(selected) < n:
                    up = xoffset
                    xoffset = (xoffset + low) / 2
                else:
                    low = xoffset
                    xoffset = (xoffset + up) / 2
            if verbose > 0:
                print("Chosen offset: {:.2f}".format(xoffset))
        else:
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset

        if plot:
            if figsize is not None:
                plt.figure(figsize=figsize)
            plt.ylim([0, 1])
            if threshold > 0:
                plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
            else:
                plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
            x = np.arange(plt.xlim()[0], plt.xlim()[1] + .1, .1)
            y = np.exp(-decay * (x - xoffset)) + yoffset
            if decay == 1:
                plt.text(.4, 0.2, "{} genes selected\ny = exp(-x+{:.2f})+{:.2f}".format(np.sum(selected), xoffset, yoffset),
                         color="k", fontsize=labelsize, transform=plt.gca().transAxes)
            else:
                plt.text(.4, 0.2, "{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}".format(np.sum(selected), decay, xoffset, yoffset),
                         color="k", fontsize=labelsize, transform=plt.gca().transAxes)

            plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
            xy = np.concatenate((np.concatenate((x[:, None], y[:, None]), axis=1), np.array([[plt.xlim()[1], 1]])))
            t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
            plt.gca().add_patch(t)
            plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
            plt.xlabel("Mean log2 nonzero expression")
            if threshold == 0:
                plt.ylabel("Frequency of zero expression")
            else:
                plt.ylabel("Frequency of near-zero expression")
            plt.tight_layout()

            if markers is not None and genes is not None:
                if markeroffsets is None:
                    markeroffsets = [(0, 0) for _ in markers]
                for num, g in enumerate(markers):
                    i = np.where(genes == g)[0]
                    plt.scatter(meanExpr[i], zeroRate[i], s=10, color="k")
                    dx, dy = markeroffsets[num]
                    plt.text(meanExpr[i] + dx + .1, zeroRate[i] + dy, g, color="k", fontsize=labelsize)
        return selected

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_SCMAGCL_MAIN_DIR = os.path.join(_ROOT_DIR, "scMAGCL-main")


def _load_local_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gcl_cfg = None
CellDataset = None
device = None
gcl_train = None
gcl_test = None


def _ensure_scMAGCL_loaded():
    """Lazy-load scMAGCL-main modules to keep --help usable without torch."""
    global gcl_cfg, CellDataset, device, gcl_train, gcl_test, torch

    if gcl_cfg is not None:
        return

    if not os.path.isdir(_SCMAGCL_MAIN_DIR):
        raise FileNotFoundError(f"scMAGCL-main not found: {_SCMAGCL_MAIN_DIR}")

    if torch is None:
        # scMAGCL-main 的 utils/scSimGCL/main 会 import torch；这里先给出清晰错误
        raise ModuleNotFoundError(
            "No module named 'torch'. Training requires PyTorch; please install torch in this environment."
        )

    gcl_utils_mod = _load_local_module("utils", os.path.join(_SCMAGCL_MAIN_DIR, "utils.py"))
    _load_local_module("scSimGCL", os.path.join(_SCMAGCL_MAIN_DIR, "scSimGCL.py"))
    gcl_config_mod = _load_local_module("config", os.path.join(_SCMAGCL_MAIN_DIR, "config.py"))
    gcl_main_mod = _load_local_module("main", os.path.join(_SCMAGCL_MAIN_DIR, "main.py"))

    gcl_cfg = gcl_config_mod.config
    CellDataset = gcl_utils_mod.CellDataset
    device = gcl_utils_mod.device
    gcl_train = gcl_main_mod.train
    gcl_test = gcl_main_mod.test


def _compat_train(**kwargs):
    _ensure_scMAGCL_loaded()
    sig = inspect.signature(gcl_train)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    out = gcl_train(**filtered)
    best_epoch, min_loss, best_z_test, best_y_test, best_x_imp_test, best_l1, best_pccs = out
    return (
        best_epoch, min_loss,
        None, best_z_test,
        None, best_y_test,
        None, best_x_imp_test,
        best_l1, best_pccs,
        None,
        None, None,
        None, None,
    )


def _compat_test(z_test_epoch, y_test_epoch, n_clusters, seed):
    _ensure_scMAGCL_loaded()
    return gcl_test([z_test_epoch], [y_test_epoch], 0, n_clusters, seed)


def _to_dense_float32(adata):
    """Convert AnnData matrix to float32 dense array."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype="float32")


def _read_modality(modality_name, h5ad_path=None, tenx_dir=None):
    """Read one modality from h5ad or 10x mtx directory."""
    if h5ad_path:
        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(f"{modality_name} h5ad not found: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)
        print(f"Loaded {modality_name} from h5ad: {h5ad_path} | shape={adata.shape}")
        return adata

    if tenx_dir:
        if not os.path.isdir(tenx_dir):
            raise FileNotFoundError(f"{modality_name} 10x directory not found: {tenx_dir}")
        try:
            adata = sc.read_10x_mtx(tenx_dir, var_names="gene_symbols", make_unique=True)
        except Exception:
            try:
                adata = sc.read_10x_mtx(tenx_dir, var_names="gene_ids", make_unique=True)
            except Exception:
                # Fallback for plain-text 10x files (matrix.mtx/features.tsv/barcodes.tsv)
                # when scanpy expects gzipped inputs in current environment.
                def _pick_existing(candidates):
                    for c in candidates:
                        p = os.path.join(tenx_dir, c)
                        if os.path.exists(p):
                            return p
                    return None

                matrix_path = _pick_existing(["matrix.mtx.gz", "matrix.mtx"])
                feature_path = _pick_existing(["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"])
                barcode_path = _pick_existing(["barcodes.tsv.gz", "barcodes.tsv"])
                if matrix_path is None or feature_path is None or barcode_path is None:
                    raise FileNotFoundError(
                        f"{modality_name} 10x directory is missing required files. "
                        "Need matrix.mtx(.gz), features/genes.tsv(.gz), barcodes.tsv(.gz)."
                    )

                mat = mmread(matrix_path).tocsr().T
                feat_df = pd.read_csv(feature_path, sep="\t", header=None)
                bc_df = pd.read_csv(barcode_path, sep="\t", header=None)

                if feat_df.shape[1] >= 2:
                    var_names = feat_df.iloc[:, 1].astype(str).values
                else:
                    var_names = feat_df.iloc[:, 0].astype(str).values
                obs_names = bc_df.iloc[:, 0].astype(str).values

                adata = sc.AnnData(X=mat)
                adata.obs_names = obs_names
                adata.var_names = var_names
                adata.var_names_make_unique()
        print(f"Loaded {modality_name} from 10x mtx: {tenx_dir} | shape={adata.shape}")
        return adata

    raise ValueError(f"{modality_name} input is missing. Provide h5ad or 10x directory.")


def _align_two_modalities(adata_atac, adata_rna):
    """Align ATAC/RNA by shared barcodes."""
    atac_barcodes = pd.Index(adata_atac.obs_names)
    rna_barcodes = pd.Index(adata_rna.obs_names)

    if len(atac_barcodes) == len(rna_barcodes) and np.array_equal(atac_barcodes.values, rna_barcodes.values):
        return adata_atac, adata_rna

    common = atac_barcodes.intersection(rna_barcodes)
    if len(common) == 0:
        raise ValueError("ATAC and RNA have no shared barcodes; cannot align modalities.")

    print(f"Barcode alignment: ATAC={len(atac_barcodes)}, RNA={len(rna_barcodes)}, shared={len(common)}")
    adata_atac = adata_atac[common].copy()
    adata_rna = adata_rna[common].copy()
    return adata_atac, adata_rna


def _load_labels(label_csv, barcodes, label_col=None):
    """Load labels from CSV and encode to integer classes."""
    label_df = pd.read_csv(label_csv)
    print(f"Label CSV columns: {label_df.columns.tolist()}")

    barcode_cols = ["barcode", "barcodes", "cell", "cell_id", "cellid", "cell_name", "Cell", "CellID"]
    barcode_col = next((c for c in barcode_cols if c in label_df.columns), None)

    preferred_cols = []
    if label_col:
        preferred_cols.append(label_col)
    preferred_cols.extend([
        "Cluster",
        "cluster",
        "cell_type",
        "celltype",
        "CellType",
        "predicted.celltype",
        "predicted_celltype",
        "label",
        "labels",
        "annotation",
        "orig.ident",
    ])
    label_col_found = next((c for c in preferred_cols if c in label_df.columns), None)
    if label_col_found is None:
        non_barcode_cols = [c for c in label_df.columns if c != barcode_col]
        if not non_barcode_cols:
            raise ValueError("Label CSV has no usable columns.")
        label_col_found = non_barcode_cols[0]
        print(f"Using first available column '{label_col_found}' as labels")

    if barcode_col is not None:
        label_series = label_df.set_index(barcode_col)[label_col_found]
        label_series = label_series.reindex(barcodes)
        missing_n = int(label_series.isna().sum())
        if missing_n > 0:
            print(f"Warning: {missing_n} barcodes have no label, filled as 'Unknown'")
            label_series = label_series.fillna("Unknown")
        labels_raw = label_series.astype(str).values
    else:
        labels_raw = label_df[label_col_found].astype(str).values
        if len(labels_raw) != len(barcodes):
            print(
                f"Warning: label length ({len(labels_raw)}) != cell count ({len(barcodes)}), "
                "will truncate or pad."
            )
            if len(labels_raw) > len(barcodes):
                labels_raw = labels_raw[: len(barcodes)]
            else:
                labels_raw = np.pad(labels_raw, (0, len(barcodes) - len(labels_raw)), constant_values="Unknown")

    y_encoded, y_classes = pd.factorize(labels_raw, sort=True)
    print(f"Loaded labels from '{label_col_found}', class count = {len(y_classes)}")
    return y_encoded.astype(np.int64), y_classes


def build_atac_rna_argparser() -> argparse.ArgumentParser:
    """
    ATAC+RNA 训练入口参数（保持原 __main__ 行为与默认值不变）。
    单独抽出来是为了复用/统一入口结构，同时不影响结果。
    """
    parser = argparse.ArgumentParser(
        description="scMAGCA preprocess -> scSimGCL (improved) training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--atac_h5ad", default=None)
    parser.add_argument("--rna_h5ad", default=None)
    parser.add_argument(
        "--atac_10x_dir",
        default=None,
        help="ATAC 10x directory with matrix.mtx/features.tsv/barcodes.tsv",
    )
    parser.add_argument(
        "--rna_10x_dir",
        default=None,
        help="RNA 10x directory with matrix.mtx/features.tsv/barcodes.tsv",
    )
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--label_col", default=None, help="Optional label column name in label_csv")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--filter1", action="store_true", default=False)
    parser.add_argument("--filter2", action="store_true", default=False)
    parser.add_argument("--f1", default=2000, type=int)
    parser.add_argument("--f2", default=2000, type=int)
    parser.add_argument("--n_clusters", type=int, default=9, help="Use -1 to auto/label estimate")
    parser.add_argument("--resolution", type=float, default=0.08)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--print_k_only", action="store_true", default=False)
    parser.add_argument("--no_clr", action="store_true", default=False)
    parser.add_argument("--no_scale", action="store_true", default=False)
    # 默认值在 main_atac_rna() 中 lazy-load gcl_cfg 后补齐，避免仅 --help 时依赖 torch。
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_cl", type=float, default=None)
    parser.add_argument("--lambda_byol", type=float, default=None)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--sweep", action="store_true", default=False, help="启用超参数扫描模式")
    parser.add_argument(
        "--sweep_param",
        type=str,
        default=None,
        help="要扫描的参数名 (lambda_cl/lambda_byol/phi1/tau)",
    )
    parser.add_argument("--sweep_values", type=str, default=None, help="扫描值,逗号分隔 (如: 0.2,0.4,0.6)")
    return parser


def main_atac_rna(argv=None):
    """
    ATAC+RNA 主入口（原 __main__ 的逻辑搬运到这里）。
    重要：不修改处理顺序/默认参数/随机性设置，确保聚类结果与历史一致。
    """
    parser = build_atac_rna_argparser()
    args = parser.parse_args(argv)
    print(args)

    if args.print_k_only and args.n_clusters > 0:
        print(f"print_k_only: using provided n_clusters={args.n_clusters}")
        sys.exit(0)

    if (args.atac_h5ad is None) == (args.atac_10x_dir is None):
        parser.error("Please provide exactly one of --atac_h5ad or --atac_10x_dir.")
    if (args.rna_h5ad is None) == (args.rna_10x_dir is None):
        parser.error("Please provide exactly one of --rna_h5ad or --rna_10x_dir.")

    _ensure_scMAGCL_loaded()
    args.seed = int(gcl_cfg["seed"]) if args.seed is None else int(args.seed)
    args.epochs = int(gcl_cfg["epochs"]) if args.epochs is None else int(args.epochs)
    args.lr = float(gcl_cfg["lr"]) if args.lr is None else float(args.lr)
    args.lambda_cl = (
        float(gcl_cfg["lambda_cl"]) if args.lambda_cl is None else float(args.lambda_cl)
    )
    args.lambda_byol = (
        float(gcl_cfg.get("lambda_byol", 1.0))
        if args.lambda_byol is None
        else float(args.lambda_byol)
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    _adata_atac = _read_modality("ATAC", h5ad_path=args.atac_h5ad, tenx_dir=args.atac_10x_dir)
    _adata_rna = _read_modality("RNA", h5ad_path=args.rna_h5ad, tenx_dir=args.rna_10x_dir)
    _adata_atac, _adata_rna = _align_two_modalities(_adata_atac, _adata_rna)

    barcodes = np.array(_adata_atac.obs_names)
    x1 = _to_dense_float32(_adata_atac)
    x2 = _to_dense_float32(_adata_rna)
    del _adata_atac, _adata_rna

    dataset_source_path = args.atac_h5ad if args.atac_h5ad is not None else args.atac_10x_dir
    dataset_name = os.path.splitext(os.path.basename(os.path.normpath(dataset_source_path)))[0]

    y = None
    if args.label_csv is not None and os.path.exists(args.label_csv):
        try:
            y, label_names = _load_labels(args.label_csv, barcodes, label_col=args.label_col)
        except Exception as e:
            print(f"⚠️  Warning: Failed to read labels from CSV: {e}")
            import traceback
            traceback.print_exc()

    # Filter constant features (variance=0) before feature selection
    # This prevents clustering failure (NMI=0, ARI=0) caused by constant features
    x1_std = x1.std(axis=0)
    x2_std = x2.std(axis=0)
    x1_nonconst_mask = x1_std > 0
    x2_nonconst_mask = x2_std > 0
    
    n_const_atac = (~x1_nonconst_mask).sum()
    n_const_rna = (~x2_nonconst_mask).sum()
    if n_const_atac > 0:
        print(f"Filtering {n_const_atac} constant features from ATAC data (out of {x1.shape[1]} total)")
        x1 = x1[:, x1_nonconst_mask]
    if n_const_rna > 0:
        print(f"Filtering {n_const_rna} constant features from RNA data (out of {x2.shape[1]} total)")
        x2 = x2[:, x2_nonconst_mask]

    if args.filter1:
        important_peaks = geneSelection(x1, n=args.f1)
        x1 = x1[:, important_peaks]
    if args.filter2:
        important_genes = geneSelection(x2, n=args.f2)
        x2 = x2[:, important_genes]

    def _preprocess_optional(adata, do_clr=True, do_scale=True, logtrans=True, normalize_input=True):
        import scipy.sparse as sp
        adata.raw = adata.copy()
        if normalize_input:
            sc.pp.normalize_total(adata)
        if do_clr:
            def seurat_clr_row(x):
                s = np.sum(np.log1p(x[x > 0]))
                expv = np.exp(s / len(x)) if len(x) > 0 else 1.0
                return np.log1p(x / (expv if expv != 0 else 1.0))
            Xraw = adata.raw.X.A if sp.issparse(adata.raw.X) else adata.raw.X
            adata.X = np.apply_along_axis(seurat_clr_row, 1, Xraw)
        if logtrans:
            sc.pp.log1p(adata)
        if do_scale:
            sc.pp.scale(adata)
        return adata

    adata1 = sc.AnnData(x1); adata1 = read_dataset(adata1, copy=True)
    if args.no_clr or args.no_scale:
        adata1 = _preprocess_optional(adata1, do_clr=(not args.no_clr), do_scale=(not args.no_scale))
    else:
        adata1 = preprocess_dataset(adata1, normalize_input=True, logtrans_input=True)
    
    # Filter constant features after preprocessing (preprocessing may create new constant features)
    import scipy.sparse as sp
    X1_processed = adata1.X.A if sp.issparse(adata1.X) else adata1.X
    X1_std = np.std(X1_processed, axis=0)
    X1_nonconst_mask = (X1_std > 0) & (~np.isnan(X1_std)) & (~np.isinf(X1_std))
    n_const_atac_post = (~X1_nonconst_mask).sum()
    if n_const_atac_post > 0:
        print(f"Filtering {n_const_atac_post} constant features from ATAC data after preprocessing (out of {X1_processed.shape[1]} total)")
        adata1 = adata1[:, X1_nonconst_mask]
    print(adata1)

    adata2 = sc.AnnData(x2); adata2 = read_dataset(adata2, copy=True)
    if args.no_clr or args.no_scale:
        adata2 = _preprocess_optional(adata2, do_clr=(not args.no_clr), do_scale=(not args.no_scale))
    else:
        adata2 = preprocess_dataset(adata2, normalize_input=True, logtrans_input=True)
    
    # Filter constant features after preprocessing
    X2_processed = adata2.X.A if sp.issparse(adata2.X) else adata2.X
    X2_std = np.std(X2_processed, axis=0)
    X2_nonconst_mask = (X2_std > 0) & (~np.isnan(X2_std)) & (~np.isinf(X2_std))
    n_const_rna_post = (~X2_nonconst_mask).sum()
    if n_const_rna_post > 0:
        print(f"Filtering {n_const_rna_post} constant features from RNA data after preprocessing (out of {X2_processed.shape[1]} total)")
        adata2 = adata2[:, X2_nonconst_mask]
    print(adata2)

    # Convert to dense arrays for concatenation
    X1_final = adata1.X.A if sp.issparse(adata1.X) else adata1.X
    X2_final = adata2.X.A if sp.issparse(adata2.X) else adata2.X
    X = np.concatenate([X1_final, X2_final], axis=1).astype('float32')
    
    # Final check: filter any constant features in concatenated matrix
    X_std = np.std(X, axis=0)
    X_nonconst_mask = (X_std > 0) & (~np.isnan(X_std)) & (~np.isinf(X_std))
    n_const_final = (~X_nonconst_mask).sum()
    if n_const_final > 0:
        print(f"Filtering {n_const_final} constant features from concatenated matrix (out of {X.shape[1]} total)")
        X = X[:, X_nonconst_mask]
    
    n_cells, n_feats = X.shape
    print(f'Concatenated feature matrix: {n_cells} cells x {n_feats} features')
    
    # Check for NaN/Inf in final matrix
    if np.isnan(X).any() or np.isinf(X).any():
        print(f"⚠️  Warning: Found NaN or Inf values in concatenated matrix!")
        print(f"   NaN count: {np.isnan(X).sum()}, Inf count: {np.isinf(X).sum()}")
        # Replace NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"   Replaced NaN/Inf with 0")

    # Check label validity
    if y is not None:
        print(f"Label info: shape={y.shape}, unique labels={len(np.unique(y))}, label range=[{y.min()}, {y.max()}]")
        if len(y) != n_cells:
            print(f"⚠️  Warning: Label length ({len(y)}) != number of cells ({n_cells})")
            print(f"   Truncating or padding labels to match cell count")
            if len(y) > n_cells:
                y = y[:n_cells]
            else:
                y = np.pad(y, (0, n_cells - len(y)), mode='constant', constant_values=0)
        if len(np.unique(y)) == 1:
            print(f"⚠️  Warning: All labels are the same! This will cause NMI=0, ARI=0")
        if len(np.unique(y)) < 2:
            print(f"⚠️  Warning: Less than 2 unique labels! Clustering metrics may be invalid")
    
    if y is not None and args.n_clusters <= 0:
        try:
            k_from_label = int(len(np.unique(y)))
            print(f"Detected labels from CSV, set n_clusters = number of unique labels = {k_from_label}")
            if args.print_k_only:
                import sys
                print(f"Estimated n_clusters from labels: {k_from_label}")
                sys.exit(0)
            args.n_clusters = k_from_label
        except Exception as e:
            print(f"Failed to infer K from labels, will fallback to Louvain. Error: {e}")

    if y is None:
        y = np.zeros((n_cells,), dtype=np.int64)

    # 设置默认save_dir为脚本所在目录的相对路径
    if args.save_dir is None:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        args.save_dir = os.path.join(_script_dir, 'results')
    
    # 立即创建save_dir文件夹
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    print(f"✅ 结果保存目录: {os.path.abspath(args.save_dir)}")
    
    if args.n_clusters <= 0:
        print("Estimating n_clusters via Louvain on concatenated ATAC+RNA features...")
        try:
            est_k = GetCluster(X, res=args.resolution, n=args.knn_k)
        except TypeError:
            est_k = GetCluster(X, args.resolution, args.knn_k)
        print(f"Estimated n_clusters: {est_k}")
        if args.print_k_only:
            import sys; sys.exit(0)
        args.n_clusters = int(est_k)

    if args.print_k_only:
        import sys
        print(f"print_k_only: final n_clusters={args.n_clusters}")
        sys.exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # 超参数扫描模式
    if args.sweep and args.sweep_param and args.sweep_values:
        sweep_values = [float(x.strip()) for x in args.sweep_values.split(',')]
        sweep_results = []
        for sv in sweep_values:
            if args.sweep_param == 'lambda_cl':
                args.lambda_cl = sv
            elif args.sweep_param == 'lambda_byol':
                args.lambda_byol = sv
            elif args.sweep_param == 'phi1':
                gcl_cfg['phi1'] = sv
            elif args.sweep_param == 'tau':
                gcl_cfg['tau'] = sv
            
            print(f"\n{'='*60}")
            print(f"扫描: {args.sweep_param}={sv}")
            print(f"{'='*60}")
            
            results_list, l1_list, pccs_list, time_list = [], [], [], []
            z_test_all = []
            y_test_all = []

            def sparse_collate(batch):
                x = [item[0] for item in batch]
                yb = torch.stack([item[1] for item in batch])
                return torch.stack(x), yb

            for run_idx in range(args.n_runs):
                run_seed = args.seed + run_idx
                X_train, X_test, y_train, y_test, bc_train, bc_test = train_test_split(
                    X, y, barcodes, test_size=0.2, random_state=run_seed, stratify=(y if len(np.unique(y))>1 else None)
                )
                train_set = CellDataset(X_train, y_train)
                test_set = CellDataset(X_test, y_test)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, 
                                                           collate_fn=sparse_collate, num_workers=0, 
                                                           pin_memory=False, persistent_workers=False)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, 
                                                          collate_fn=sparse_collate, num_workers=0,
                                                          pin_memory=False, persistent_workers=False)
                input_dim = n_feats

                t0 = time()
                best_epoch, min_loss, z_train_epoch, z_test_epoch, y_train_epoch, y_test_epoch, x_imp_train_epoch, x_imp_test_epoch, best_l1, best_pccs, _, clisi, casw, z_all_epoch, y_all_epoch = _compat_train(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    input_dim=input_dim,
                    graph_head=gcl_cfg['graph_head'],
                    phi=gcl_cfg['phi'],
                    gcn_dim=gcl_cfg['gcn_dim'],
                    mlp_dim=gcl_cfg['mlp_dim'],
                    prob_feature=gcl_cfg['prob_feature'],
                    prob_edge=gcl_cfg['prob_edge'],
                    tau=gcl_cfg['tau'],
                    alpha=gcl_cfg['alpha'],
                    beta=gcl_cfg['beta'],
                    lambda_cl=args.lambda_cl,
                    dropout=gcl_cfg['dropout'],
                    lr=args.lr,
                    seed=run_seed,
                    epochs=args.epochs,
                    device=device,
                    knn_k=gcl_cfg.get('knn_k', 15),
                    phi1=gcl_cfg['phi1'],
                    lambda_byol=args.lambda_byol,
                )
                train_time = time() - t0
                time_list.append(train_time)
                
                results = _compat_test(z_test_epoch, y_test_epoch, args.n_clusters, run_seed)
                print(f"  Run{run_idx+1}: CA={results['CA']:.4f}, NMI={results['NMI']:.4f}, ARI={results['ARI']:.4f}, Loss={min_loss:.6f}, Time={train_time:.1f}s")
                results_list.append(results); l1_list.append(best_l1); pccs_list.append(best_pccs)

                z_test_all.append(np.vstack(z_test_epoch))
                y_test_all.append(np.hstack(y_test_epoch))

                # ========== sweep模式：五段式彻底内存清理 ==========
                print(f"🧹 开始清理sweep run {run_idx+1}的内存...")
                
                # 阶段1：关闭DataLoader
                try:
                    del train_loader._iterator
                except:
                    pass
                try:
                    del test_loader._iterator
                except:
                    pass
                del train_loader, test_loader
                
                # 阶段2：清理Dataset和分割数据
                del train_set, test_set
                del X_train, X_test, y_train, y_test, bc_train, bc_test
                
                # 阶段3：清理Results对象
                del results
                
                # 阶段4：清理Embeddings和标量
                del z_train_epoch, z_test_epoch, y_train_epoch, y_test_epoch
                del x_imp_train_epoch, x_imp_test_epoch
                del best_l1, best_pccs, clisi, casw
                del z_all_epoch, y_all_epoch
                del best_epoch, min_loss, train_time
                
                # 阶段5：多轮强制垃圾回收
                for _ in range(3):
                    gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                # 打印内存状态
                if psutil is not None:
                    process = psutil.Process()
                    mem_mb = process.memory_info().rss / 1024 / 1024
                    print(f"✅ Sweep run {run_idx+1}清理完成，当前内存: {mem_mb:.2f} MB")
                # ========== END sweep模式内存清理 ==========
            
            # 计算平均值
            ca_mean = np.mean([r['CA'] for r in results_list])
            nmi_mean = np.mean([r['NMI'] for r in results_list])
            ari_mean = np.mean([r['ARI'] for r in results_list])
            time_mean = np.mean(time_list)
            time_std = np.std(time_list) if len(time_list) > 1 else 0
            
            # 合并所有runs的embeddings用于可视化
            z_test_combined = np.vstack(z_test_all) if len(z_test_all) > 0 else np.array([])
            y_test_combined = np.hstack(y_test_all) if len(y_test_all) > 0 else np.array([])
            
            sweep_results.append({
                'param': args.sweep_param,
                'value': sv,
                'CA': ca_mean,
                'NMI': nmi_mean,
                'ARI': ari_mean,
                'time_sec': time_mean,
                'time_std': time_std
            })
            
        # 保存扫描结果为CSV
        sweep_df = pd.DataFrame(sweep_results)
        sweep_csv = os.path.join(args.save_dir, f'sweep_{args.sweep_param}_results.csv')
        sweep_df.to_csv(sweep_csv, index=False)
        
        # 扫描模式的最终汇总统计
        print("\n" + "="*60)
        print(f"超参数扫描总结: {args.sweep_param}")
        print("="*60)
        print(f"扫描值数量: {len(sweep_results)}")
        print(f"每个值运行次数: {args.n_runs}")
        print(f"总实验次数: {len(sweep_results) * args.n_runs}")
        
        # 统计各指标的最大值及对应的参数值
        ca_scores = sweep_df['CA'].values
        nmi_scores = sweep_df['NMI'].values
        ari_scores = sweep_df['ARI'].values
        time_vals = sweep_df['time_sec'].values
        
        best_ca_idx = np.argmax(ca_scores)
        best_nmi_idx = np.argmax(nmi_scores)
        best_ari_idx = np.argmax(ari_scores)
        
        print(f"\n最优结果:")
        print(f"  CA最优值: {ca_scores[best_ca_idx]:.4f} @ {args.sweep_param}={sweep_df.iloc[best_ca_idx]['value']}")
        print(f"  NMI最优值: {nmi_scores[best_nmi_idx]:.4f} @ {args.sweep_param}={sweep_df.iloc[best_nmi_idx]['value']}")
        print(f"  ARI最优值: {ari_scores[best_ari_idx]:.4f} @ {args.sweep_param}={sweep_df.iloc[best_ari_idx]['value']}")
        print(f"\n平均性能:")
        print(f"  CA  : {np.mean(ca_scores):.4f} ± {np.std(ca_scores):.4f}")
        print(f"  NMI : {np.mean(nmi_scores):.4f} ± {np.std(nmi_scores):.4f}")
        print(f"  ARI : {np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}")
        print(f"  耗时: {np.mean(time_vals):.1f} ± {np.mean(sweep_df['time_std'].values):.1f} 秒")
        print("="*60)
        print(f"\n✅ 扫描完成！结果已保存到: {sweep_csv}")
        print(sweep_df.to_string(index=False))
        
        print("="*60)
        
    else:
        # 常规单次运行模式
        results_list, l1_list, pccs_list = [], [], []

        def sparse_collate(batch):
            x = [item[0] for item in batch]
            yb = torch.stack([item[1] for item in batch])
            return torch.stack(x), yb

        for run_idx in range(args.n_runs):
            run_seed = args.seed + run_idx
            X_train, X_test, y_train, y_test, bc_train, bc_test = train_test_split(
                X, y, barcodes, test_size=0.2, random_state=run_seed, stratify=(y if len(np.unique(y))>1 else None)
            )
            train_set = CellDataset(X_train, y_train)
            test_set = CellDataset(X_test, y_test)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True,
                                                       collate_fn=sparse_collate, num_workers=0,
                                                       pin_memory=False, persistent_workers=False)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False,
                                                      collate_fn=sparse_collate, num_workers=0,
                                                      pin_memory=False, persistent_workers=False)
            input_dim = n_feats

            print(f"Start scSimGCL (improved) training on concatenated ATAC+RNA features... run {run_idx+1}/{args.n_runs}")
            t0 = time()
            best_epoch, min_loss, z_train_epoch, z_test_epoch, y_train_epoch, y_test_epoch, x_imp_train_epoch, x_imp_test_epoch, best_l1, best_pccs, _, clisi, casw, z_all_epoch, y_all_epoch = _compat_train(
                train_loader=train_loader,
                test_loader=test_loader,
                input_dim=input_dim,
                graph_head=gcl_cfg['graph_head'],
                phi=gcl_cfg['phi'],
                gcn_dim=gcl_cfg['gcn_dim'],
                mlp_dim=gcl_cfg['mlp_dim'],
                prob_feature=gcl_cfg['prob_feature'],
                prob_edge=gcl_cfg['prob_edge'],
                tau=gcl_cfg['tau'],
                alpha=gcl_cfg['alpha'],
                beta=gcl_cfg['beta'],
                lambda_cl=args.lambda_cl,
                dropout=gcl_cfg['dropout'],
                lr=args.lr,
                seed=run_seed,
                epochs=args.epochs,
                device=device,
                knn_k=gcl_cfg.get('knn_k', 15),
                phi1=gcl_cfg['phi1'],
                lambda_byol=args.lambda_byol,
            )
            print('Training time: %d seconds.' % int(time() - t0))

            results = _compat_test(z_test_epoch, y_test_epoch, args.n_clusters, run_seed)
            print(f"Results (run {run_idx+1}): CA={results['CA']:.4f}, NMI={results['NMI']:.4f}, ARI={results['ARI']:.4f}")
            results_list.append(results); l1_list.append(best_l1); pccs_list.append(best_pccs)

            # ========== 五段式彻底内存清理 ==========
            print(f"🧹 开始清理run {run_idx+1}的内存...")

            # 阶段1：关闭DataLoader
            try:
                del train_loader._iterator
            except:
                pass
            try:
                del test_loader._iterator
            except:
                pass
            del train_loader, test_loader

            # 阶段2：清理Dataset和分割数据
            del train_set, test_set
            del X_train, X_test, y_train, y_test, bc_train, bc_test

            # 阶段3：Results对象清理
            del results

            # 阶段4：Embeddings和所有训练返回值清理
            del z_train_epoch, z_test_epoch, y_train_epoch, y_test_epoch
            del x_imp_train_epoch, x_imp_test_epoch
            del best_l1, best_pccs, clisi, casw
            del z_all_epoch, y_all_epoch
            del best_epoch, min_loss

            # 阶段5：多轮强制垃圾回收和CUDA清理
            for _ in range(3):
                gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            if psutil is not None:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                print(f"✅ Run {run_idx+1}清理完成，当前内存: {mem_mb:.2f} MB")
            # ========== END 内存清理 ==========

        if args.n_runs > 1:
            ca_vals = [r['CA'] for r in results_list]
            nmi_vals = [r['NMI'] for r in results_list]
            ari_vals = [r['ARI'] for r in results_list]
            print("="*60)
            print(f"Runs: {args.n_runs}, seed base: {args.seed}")
            print(f"CA  : {np.mean(ca_vals):.4f} ± {np.std(ca_vals):.4f}")
            print(f"NMI : {np.mean(nmi_vals):.4f} ± {np.std(nmi_vals):.4f}")
            print(f"ARI : {np.mean(ari_vals):.4f} ± {np.std(ari_vals):.4f}")
            if len(l1_list) > 0:
                print(f"L1  : {float(np.nanmean(l1_list)):.6f} ± {float(np.nanstd(l1_list)):.6f}")
            if len(pccs_list) > 0:
                print(f"PCCS: {float(np.nanmean(pccs_list)):.4f} ± {float(np.nanstd(pccs_list)):.4f}")


if __name__ == "__main__":
    main_atac_rna()

