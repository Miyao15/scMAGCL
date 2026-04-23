#!/usr/bin/env python3
"""
Baseline Preprocessing: Concatenation of Normalized and HVG-selected RNA and ADT data.
Pipeline:
1) Load RNA/ADT data (supports H5AD, 10X MTX, CSV).
2) Align barcodes via intersection.
3) RNA preprocessing: normalize_total -> log1p -> highly_variable_genes -> subset -> scale.
4) ADT preprocessing: normalize_total -> log1p -> scale.
5) Feature concatenation and H5 export.
6) Optional: Direct execution of training via scMAGCL main module.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import argparse
import re
from scipy.io import mmread
from scipy.sparse import csr_matrix

# Dynamic path configuration for scMAGCL modules
_PREP_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.join(os.path.dirname(_PREP_DIR), "scMAGCL-main")
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

def _robust_mmread(matrix_path: str):
    """Reads MTX files with error handling for header mismatches."""
    try:
        return mmread(matrix_path).tocsr()
    except ValueError as e:
        print(f"Warning: MTX parsing failed, attempting repair for {matrix_path}")
        with open(matrix_path, 'r') as f:
            lines = f.readlines()
        
        header_idx = None
        header_lines = []
        for i, line in enumerate(lines):
            if line.startswith('%%') or line.startswith('%'):
                header_lines.append(line)
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                header_idx = i
                declared_rows, declared_cols, _ = map(int, parts)
                break
        
        if header_idx is None:
            raise ValueError(f"Unable to parse MTX header: {matrix_path}")
        
        data_lines = lines[header_idx + 1:]
        valid_data_lines = []
        max_row, max_col = 0, 0
        
        for line in data_lines:
            stripped = line.strip()
            if not stripped: continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    row_idx, col_idx = int(parts[0]), int(parts[1])
                    valid_data_lines.append(line)
                    max_row, max_col = max(max_row, row_idx), max(max_col, col_idx)
                except ValueError:
                    continue
        
        actual_entries = len(valid_data_lines)
        actual_rows = max(declared_rows, max_row)
        actual_cols = max(declared_cols, max_col)
        
        fixed_lines = header_lines.copy()
        fixed_lines.append(f"{actual_rows} {actual_cols} {actual_entries}\n")
        fixed_lines.extend(valid_data_lines)
        
        temp_path = matrix_path + ".fixed.mtx"
        with open(temp_path, 'w') as f:
            f.writelines(fixed_lines)
        return mmread(temp_path).tocsr()

def _read_label_csv_flexible(label_csv_path: str):
    """Loads labels from CSV with flexible column detection and string enforcement."""
    if not os.path.exists(label_csv_path): return None
    try:
        df = pd.read_csv(label_csv_path)
    except Exception:
        df = pd.read_csv(label_csv_path, header=None)
        
    if 'Barcode' in df.columns and 'Cluster' in df.columns:
        out = df[['Barcode', 'Cluster']].copy()
    elif 'cell' in df.columns and 'cluster' in df.columns:
        out = df[['cell', 'cluster']].copy()
        out.columns = ['Barcode', 'Cluster']
    else:
        out = df.iloc[:, :2].copy()
        out.columns = ['Barcode', 'Cluster']
    
    out['Barcode'] = out['Barcode'].astype(str)
    out['Cluster'] = out['Cluster'].astype(str)
    return out

def _normalize_barcode(bc: str) -> str:
    return str(bc).strip().replace('"', '').replace("'", "").upper()

def _strip_suffix_after_dash(bc: str) -> str:
    return _normalize_barcode(bc).split('-')[0]

def _core16(bc: str) -> str:
    s = _normalize_barcode(bc)
    m = re.findall(r"[ACGT]{16,}", s)
    return m[0][:16] if m else ""

def _build_barcode_to_label(labels_df: pd.DataFrame):
    """Maps normalized barcodes to their corresponding labels."""
    mapping = {}
    if labels_df is None or labels_df.empty: return mapping
    for bc, lab in zip(labels_df['Barcode'], labels_df['Cluster']):
        bc_norm = _normalize_barcode(bc)
        bc_base = _strip_suffix_after_dash(bc_norm)
        bc_c16  = _core16(bc_norm)
        mapping[bc_norm] = lab
        mapping[bc_base] = lab
        if bc_c16: mapping[bc_c16] = lab
    return mapping

def _preprocess_rna(rna: sc.AnnData, filter_n_genes: int = None, skip_normalize: bool = False,
                    skip_scale: bool = False, hvg_min_mean: float = 0.0125,
                    hvg_max_mean: float = 3, hvg_min_disp: float = 0.5):
    """Standard scRNA-seq preprocessing workflow."""
    rna = rna.copy()
    print(f"RNA Preprocessing - Initial shape: {rna.shape}")
    
    if not skip_normalize: sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    
    if filter_n_genes and filter_n_genes > 0:
        X_array = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
        var_per_gene = np.var(X_array, axis=0)
        top_gene_idx = np.sort(np.argsort(var_per_gene)[::-1][:filter_n_genes])
        rna = rna[:, top_gene_idx].copy()
    else:
        sc.pp.highly_variable_genes(rna, min_mean=hvg_min_mean, max_mean=hvg_max_mean, min_disp=hvg_min_disp)
        rna = rna[:, rna.var.highly_variable].copy()
    
    if not skip_scale: sc.pp.scale(rna)
    
    X_array = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    if np.any(~np.isfinite(X_array)):
        rna.X = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"RNA Preprocessing - Final shape: {rna.shape}")
    return rna

def main():
    parser = argparse.ArgumentParser(description="Baseline RNA+ADT Preprocessing")
    parser.add_argument("--rna_10x_dir", type=str, default=None)
    parser.add_argument("--adt_csv", type=str, default=None)
    parser.add_argument("--rna_h5ad", type=str, default=None)
    parser.add_argument("--adt_h5ad", type=str, default=None)
    parser.add_argument("--label_csv", type=str, default=None)
    parser.add_argument("--tag", type=str, default="4")
    parser.add_argument("--out_dir", type=str, default="scMAGCL_data")
    parser.add_argument("--filter2", "-f2", type=int, default=None)
    parser.add_argument("--no_clr", action="store_true")
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--n_runs", type=int, default=1)
    args = parser.parse_args()

    print("Executing RNA+ADT feature concatenation pipeline...")

    # Load RNA
    if args.rna_h5ad and os.path.exists(args.rna_h5ad):
        rna = sc.read_h5ad(args.rna_h5ad)
    elif args.rna_10x_dir:
        rna = sc.read_10x_mtx(args.rna_10x_dir, var_names='gene_symbols', make_unique=True)
    else:
        raise ValueError("Invalid RNA input path.")
    rna.var_names_make_unique()

    # Load ADT
    if args.adt_h5ad and os.path.exists(args.adt_h5ad):
        adt = sc.read_h5ad(args.adt_h5ad)
    elif args.adt_csv:
        df_adt = pd.read_csv(args.adt_csv, index_col=0)
        adt = sc.AnnData(df_adt.values, obs={'obs_names': df_adt.index}, var={'var_names': df_adt.columns})
    else:
        raise ValueError("Invalid ADT input path.")
    adt.var_names_make_unique()

    print(f"Initial dimensions - RNA: {rna.shape}, ADT: {adt.shape}")

    # Map RNA indices to ADT names if numeric barcodes are detected
    if all(bc.isdigit() for bc in rna.obs_names):
        print("Warning: Numeric barcodes detected. Mapping RNA indices to ADT names.")
        if rna.shape[0] == adt.shape[0]:
            rna.obs_names = adt.obs_names.copy()
        else:
            raise ValueError("Cell count mismatch between RNA and ADT.")

    # Align barcodes
    rna.obs_names = pd.Index([_strip_suffix_after_dash(bc) for bc in rna.obs_names])
    adt.obs_names = pd.Index([_strip_suffix_after_dash(bc) for bc in adt.obs_names])
    rna.obs_names_make_unique()
    adt.obs_names_make_unique()
    
    intersect = np.array(sorted(rna.obs_names.intersection(adt.obs_names)))
    if len(intersect) == 0:
        raise ValueError("No overlapping barcodes found.")
    
    rna = rna[intersect, :].copy()
    adt = adt[intersect, :].copy()
    print(f"Aligned cell count: {rna.shape[0]}")

    # Process modalities
    rna = _preprocess_rna(rna, filter_n_genes=args.filter2, skip_normalize=args.no_clr, skip_scale=args.no_scale)
    if not args.no_clr: sc.pp.normalize_total(adt, target_sum=1e4)
    sc.pp.log1p(adt)
    if not args.no_scale: sc.pp.scale(adt, max_value=10)

    # Concatenate and Save
    X_rna = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    X_adt = adt.X.toarray() if hasattr(adt.X, 'toarray') else adt.X
    X_concat = np.nan_to_num(np.hstack([X_rna, X_adt])).astype(np.float32)
    
    os.makedirs(args.out_dir, exist_ok=True)
    h5_path = os.path.join(args.out_dir, f'baseline_scMAGCL_input_{args.tag}.h5')
    
    labels_df_src = _read_label_csv_flexible(args.label_csv)
    if labels_df_src is not None:
        mapping = _build_barcode_to_label(labels_df_src)
        raw_labels = [mapping.get(_strip_suffix_after_dash(bc), "Unknown") for bc in rna.obs_names]
        unique_labels = sorted(list(set(raw_labels)))
        y = np.array([unique_labels.index(lab) for lab in raw_labels], dtype=int)
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('X', data=X_concat, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            f.create_dataset('cell_barcodes', data=[s.encode('utf-8') for s in rna.obs_names])
            f.create_dataset('label_names', data=[str(s).encode('utf-8') for s in unique_labels])
    else:
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('X', data=X_concat, compression='gzip')
            f.create_dataset('y', data=np.zeros(X_concat.shape[0], dtype=int), compression='gzip')

    print(f"Dataset successfully saved to: {h5_path}")

    # Direct Training Integration
    if args.train:
        print("Initializing training procedure...")
        from config import config
        from utils import loader_construction, device
        from main import train, test
        
        for i in range(args.n_runs):
            cur_seed = config['seed'] + i
            train_loader, test_loader, input_dim = loader_construction(h5_path)
            
            # Match the 4 return values of the updated train() function in main.py
            best_epoch, min_loss, z_test, y_test = train(
                train_loader, test_loader, input_dim, config['graph_head'], config['phi'], 
                config['gcn_dim'], config['mlp_dim'], config['prob_feature'], config['prob_edge'],
                config['tau'], config['alpha'], config['beta'], config['lambda_cl'], 
                config['dropout'], config['lr'], cur_seed, config['epochs'], device,
                phi1=config['phi1'], lambda_byol=config['lambda_byol']
            )
            
            results = test(z_test, y_test, args.n_clusters or 9, cur_seed)
            print(f"Run {i+1} results: CA={results['CA']:.4f}, NMI={results['NMI']:.4f}, ARI={results['ARI']:.4f}")

if __name__ == "__main__":
    main()