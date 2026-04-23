#!/usr/bin/env python3
"""
基线对照：直接拼接归一化+HVG处理后的RNA与ADT
流程：
1) 读取RNA/ADT (兼容 H5AD, 10X, CSV)
2) 对齐细胞条形码的交集（同序）
3) RNA：normalize_total → log1p → highly_variable_genes → 子集 → scale
4) ADT：normalize_total → log1p → scale（不做HVG）
5) 横向拼接特征并保存 H5 文件
6) 可选：自动调用 scMAGCL-main/main.py 进行训练评估
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from datetime import datetime
import argparse
from scipy.io import mmread
from scipy.sparse import csr_matrix
import re

# 动态将 scMAGCL-main 目录加入系统路径，以便后续直接导入 main.py
_PREP_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.join(os.path.dirname(_PREP_DIR), "scMAGCL-main")
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

def _robust_mmread(matrix_path: str):
    try:
        return mmread(matrix_path).tocsr()
    except ValueError as e:
        error_msg = str(e)
        if "'entries' in header is smaller than" in error_msg or \
           "not enough values to unpack" in error_msg or \
           "index exceeds matrix dimensions" in error_msg:
            print(f"  ⚠️ MTX 文件存在问题，尝试修复...")
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
                    declared_rows, declared_cols, declared_entries = map(int, parts)
                    break
            
            if header_idx is None:
                raise ValueError(f"无法解析 MTX 文件头部: {matrix_path}")
            
            data_lines = lines[header_idx + 1:]
            valid_data_lines = []
            invalid_count, max_row, max_col = 0, 0, 0
            
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
                        invalid_count += 1
                else:
                    invalid_count += 1
            
            actual_entries = len(valid_data_lines)
            actual_rows = max(declared_rows, max_row)
            actual_cols = max(declared_cols, max_col)
            
            dimension_changed = (actual_rows != declared_rows or actual_cols != declared_cols)
            entries_changed = (actual_entries != declared_entries)
            
            if dimension_changed or entries_changed or invalid_count > 0:
                fixed_lines = header_lines.copy()
                fixed_lines.append(f"{actual_rows} {actual_cols} {actual_entries}\n")
                fixed_lines.extend(valid_data_lines)
                temp_path = matrix_path + ".fixed.mtx"
                with open(temp_path, 'w') as f:
                    f.writelines(fixed_lines)
                try:
                    return mmread(temp_path).tocsr()
                except Exception as e2:
                    raise ValueError(f"修复后仍无法读取 MTX 文件: {e2}")
            else:
                raise e
        else:
            raise e


def _read_label_csv_flexible(label_csv_path: str):
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
        if df.shape[1] >= 2:
            out = df.iloc[:, :2].copy()
            out.columns = ['Barcode', 'Cluster']
        elif df.shape[1] == 1:
            out = df.copy().reset_index().iloc[:, :2]
            out.columns = ['Barcode', 'Cluster']
        else:
            df2 = pd.read_csv(label_csv_path, header=None)
            if df2.shape[1] >= 2:
                out = df2.iloc[:, :2].copy()
                out.columns = ['Barcode', 'Cluster']
            else:
                raise ValueError("无法解析标签CSV，请提供两列（Barcode, Cluster）。")
    out['Barcode'] = out['Barcode'].astype(str)
    return out

def _normalize_barcode(bc: str) -> str:
    if bc is None: return ""
    return str(bc).strip().replace('"', '').replace("'", "").upper()

def _strip_suffix_after_dash(bc: str) -> str:
    s = _normalize_barcode(bc)
    return s.split('-')[0]

def _core16(bc: str) -> str:
    s = _normalize_barcode(bc)
    m = re.findall(r"[ACGT]{16,}", s)
    return m[0][:16] if m else ""

def _build_barcode_to_label(labels_df: pd.DataFrame):
    mapping = {}
    if labels_df is None or labels_df.empty: return mapping
    for bc, lab in zip(labels_df['Barcode'], labels_df['Cluster']):
        bc_norm = _normalize_barcode(bc)
        bc_base = _strip_suffix_after_dash(bc_norm)
        bc_c16  = _core16(bc_norm)
        if bc_norm not in mapping: mapping[bc_norm] = lab
        if bc_base not in mapping: mapping[bc_base] = lab
        if bc_c16 and bc_c16 not in mapping: mapping[bc_c16] = lab
    return mapping


def _preprocess_rna(rna: sc.AnnData, filter_n_genes: int = None, skip_normalize: bool = False,
                    skip_scale: bool = False, hvg_min_mean: float = 0.0125,
                    hvg_max_mean: float = 3, hvg_min_disp: float = 0.5):
    rna = rna.copy()
    print(f"  [RNA] 原始: {rna.shape}")
    X_array = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    var_per_gene = np.var(X_array, axis=0)
    if np.sum(~(var_per_gene > 0)) > 0:
        rna = rna[:, var_per_gene > 0].copy()
    if not skip_normalize: sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    if filter_n_genes is not None and filter_n_genes > 0:
        X_array = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
        var_per_gene = np.var(X_array, axis=0)
        top_gene_idx = np.sort(np.argsort(var_per_gene)[::-1][:filter_n_genes])
        rna = rna[:, top_gene_idx].copy()
    else:
        sc.pp.highly_variable_genes(rna, min_mean=hvg_min_mean, max_mean=hvg_max_mean, min_disp=hvg_min_disp)
        rna = rna[:, rna.var.highly_variable].copy()
    if not skip_scale: sc.pp.scale(rna)
    X_array = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    var_per_gene = np.var(X_array, axis=0)
    if np.sum(~(var_per_gene > 0)) > 0: rna = rna[:, var_per_gene > 0].copy()
    X_array = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    if np.sum(np.isnan(X_array)) > 0 or np.sum(np.isinf(X_array)) > 0:
        rna.X = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [RNA] 最终: {rna.shape}")
    return rna


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_10x_dir", type=str, default=None)
    parser.add_argument("--adt_csv", type=str, default=None)
    parser.add_argument("--rna_h5ad", type=str, default=None, help="RNA数据的h5ad格式文件路径")
    parser.add_argument("--adt_h5ad", type=str, default=None, help="ADT数据的h5ad格式文件路径")
    parser.add_argument("--label_csv", type=str, default=None)
    parser.add_argument("--tag", type=str, default="4")
    parser.add_argument("--out_dir", type=str, default="scSimGCL_data")
    parser.add_argument("--filter2", "-f2", type=int, default=None)
    parser.add_argument("--no_clr", action="store_true")
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument("--hvg_min_mean", type=float, default=0.0125)
    parser.add_argument("--hvg_max_mean", type=float, default=3)
    parser.add_argument("--hvg_min_disp", type=float, default=0.5)
    
    # 自动训练相关参数
    parser.add_argument("--train", action="store_true", help="数据拼接完成后，直接送入 main.py 训练")
    parser.add_argument("--n_clusters", type=int, default=None, help="KMeans 聚类数")
    parser.add_argument("--n_runs", type=int, default=1, help="连续跑几次实验取平均")
    args = parser.parse_args()

    print("=" * 60)
    print("基线：拼接归一化+HVG后的RNA与ADT (支持 H5AD 格式)")
    print("=" * 60)

    # 1. 载入 RNA 数据
    if args.rna_h5ad and os.path.exists(args.rna_h5ad):
        print(f"  --> 读取 RNA H5AD: {args.rna_h5ad}")
        rna = sc.read_h5ad(args.rna_h5ad)
    elif args.rna_10x_dir:
        def find_in_dir(d, pat_list):
            for pat in pat_list:
                fs = glob.glob(os.path.join(d, pat))
                if fs: return fs[0]
            return None
        matrix_path = find_in_dir(args.rna_10x_dir, ["*matrix.mtx*", "matrix.mtx*"])
        barcodes_path = find_in_dir(args.rna_10x_dir, ["*barcodes.tsv*", "barcodes.tsv*"])
        features_path = find_in_dir(args.rna_10x_dir, ["*features.tsv*", "genes.tsv*"])
        X_coo = _robust_mmread(matrix_path)
        barcodes = pd.read_csv(barcodes_path, header=None, sep='\t').iloc[:, 0].astype(str).values
        feat_df = pd.read_csv(features_path, header=None, sep='\t')
        gene_names = feat_df.iloc[:, 1 if feat_df.shape[1] >= 2 else 0].astype(str).values
        n_cells = len(barcodes)
        if X_coo.shape == (n_cells, len(gene_names)): X_rna = X_coo
        elif X_coo.shape == (len(gene_names), n_cells): X_rna = X_coo.T
        rna = sc.AnnData(X_rna)
        rna.obs_names = pd.Index(barcodes)
        rna.var_names = pd.Index(gene_names)
    else:
        raise ValueError("请提供有效的 --rna_h5ad 或 --rna_10x_dir 路径！")
    rna.var_names_make_unique()

    # 2. 载入 ADT 数据
    if args.adt_h5ad and os.path.exists(args.adt_h5ad):
        print(f"  --> 读取 ADT H5AD: {args.adt_h5ad}")
        adt = sc.read_h5ad(args.adt_h5ad)
    elif args.adt_csv:
        df_adt = pd.read_csv(args.adt_csv, index_col=0)
        df_adt.index = df_adt.index.astype(str)
        df_adt.columns = df_adt.columns.astype(str)
        if df_adt.shape[0] == rna.shape[0]: df_cells_first = df_adt
        elif df_adt.shape[1] == rna.shape[0]: df_cells_first = df_adt.T
        else: df_cells_first = df_adt
        adt = sc.AnnData(df_cells_first.values)
        adt.obs_names = pd.Index(df_cells_first.index)
        adt.var_names = pd.Index(df_cells_first.columns)
    else:
        raise ValueError("请提供有效的 --adt_h5ad 或 --adt_csv 路径！")
    adt.var_names_make_unique()

    labels_df_src = _read_label_csv_flexible(args.label_csv) if args.label_csv else None

    print(f"✓ RNA 初始维度: {rna.shape}")
    print(f"✓ ADT 初始维度: {adt.shape}")

    if all(bc.isdigit() for bc in rna.obs_names):
        print("\n💡 【严重警告修复】检测到 RNA 的 barcodes 只是纯数字索引 (0, 1, 2...)！")
        if rna.shape[0] == adt.shape[0]:
            print("   👉 RNA 细胞数与 ADT 完美一致！正在将 ADT 的真实细胞名字强行赋予 RNA...")
            rna.obs_names = adt.obs_names.copy()
        else:
            raise ValueError(f"RNA细胞数({rna.shape[0]})和ADT细胞数({adt.shape[0]})不一致，无法对齐！")

    # 3. 对齐细胞条形码
    print("\n[2/4] 统一条形码并对齐交集...")
    rna.obs_names = pd.Index([_strip_suffix_after_dash(bc) for bc in rna.obs_names])
    adt.obs_names = pd.Index([_strip_suffix_after_dash(bc) for bc in adt.obs_names])
    rna.obs_names_make_unique()
    adt.obs_names_make_unique()
    
    intersect = rna.obs_names.intersection(adt.obs_names)
    if len(intersect) == 0:
        raise ValueError("去除了后缀后，RNA与ADT仍没有共同的细胞条形码，无法拼接。")
    intersect = np.array(sorted(intersect))
    rna = rna[intersect, :].copy()
    adt = adt[intersect, :].copy()
    print(f"✓ 对齐后细胞数: {rna.shape[0]}")

    # 4. 预处理
    print("\n[3/4] 开始预处理...")
    rna = _preprocess_rna(rna, filter_n_genes=args.filter2, skip_normalize=args.no_clr, skip_scale=args.no_scale, hvg_min_mean=args.hvg_min_mean, hvg_max_mean=args.hvg_max_mean, hvg_min_disp=args.hvg_min_disp)
    if not args.no_clr: sc.pp.normalize_total(adt, target_sum=1e4)
    sc.pp.log1p(adt)
    if not args.no_scale: sc.pp.scale(adt, max_value=10)
    print(f"✓ RNA(预处理后): {rna.shape} | ADT(标准化后): {adt.shape}")

    # 5. 拼接保存
    print("\n[4/4] 拼接保存...")
    X_rna = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    X_adt = adt.X.toarray() if hasattr(adt.X, 'toarray') else adt.X
    X_concat = np.asarray(np.hstack([X_rna, X_adt]), dtype=np.float32)
    
    if np.isnan(X_concat).any() or np.isinf(X_concat).any():
        X_concat = np.nan_to_num(X_concat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
    os.makedirs(args.out_dir, exist_ok=True)
    h5_path = os.path.join(args.out_dir, f'baseline_scSimGCL_input_{args.tag}.h5')
    
    unique_labels_count = args.n_clusters if args.n_clusters else 9

    if labels_df_src is not None:
        barcode_to_label = _build_barcode_to_label(labels_df_src)
        raw_labels = [barcode_to_label.get(_strip_suffix_after_dash(bc), None) for bc in rna.obs_names]
        non_null_vals = [v for v in raw_labels if v is not None]
        unique_labels = sorted(list({str(v) for v in non_null_vals}))
        label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
        y = np.array([label_to_int.get(str(v), -1) if v is not None else -1 for v in raw_labels], dtype=int)
        label_names = np.array(unique_labels, dtype=object)
        unique_labels_count = len(unique_labels)
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('X', data=X_concat, compression='gzip', compression_opts=9)
            f.create_dataset('y', data=y, compression='gzip', compression_opts=9)
            f.create_dataset('cell_barcodes', data=[s.encode('utf-8') for s in rna.obs_names], compression='gzip')
            f.create_dataset('label_names', data=[s.encode('utf-8') for s in label_names], compression='gzip')
            f.attrs['n_cells'] = X_concat.shape[0]
            f.attrs['n_clusters'] = len(unique_labels)
        print(f"✓ H5完美生成，包含真实的细胞名字！路径: {h5_path}")
    else:
        # 如果没有传标签，就造一个全 0 标签，以便能写入符合 loader_construction 标准的 H5
        print("⚠️ 未提供 label_csv：写入无标签的默认 H5。")
        y = np.zeros(X_concat.shape[0], dtype=int)
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('X', data=X_concat, compression='gzip', compression_opts=9)
            f.create_dataset('y', data=y, compression='gzip', compression_opts=9)

    # 6. 直接调用 main.py 进行训练
    if args.train:
        print("\n" + "=" * 60)
        print("🚀 数据拼接完毕！正在直接启动 main.py 训练流程...")
        print("=" * 60)
        
        try:
            from config import config
            from utils import loader_construction, device
            from main import train, test
        except ImportError as e:
            print(f"❌ 导入主模型失败，请确认当前系统路径能找到 scMAGCL-main。报错信息：{e}")
            return
            
        target_k = args.n_clusters if args.n_clusters is not None else unique_labels_count

        for i in range(args.n_runs):
            cur_seed = config['seed'] + i
            print(f"\n--- 第 {i+1}/{args.n_runs} 次实验 (Seed: {cur_seed}) ---")
            
            # 这里的 loader_construction 就是 utils.py 里的函数，专门读取咱们刚才生成的那个 H5
            train_loader, test_loader, input_dim = loader_construction(h5_path)
            
            best_epoch, min_loss, best_z_test, best_y_test, best_x_imp_test, best_l1, best_pccs = train(
                train_loader=train_loader, 
                test_loader=test_loader, 
                input_dim=input_dim, 
                graph_head=config['graph_head'], 
                phi=config['phi'], 
                gcn_dim=config['gcn_dim'], 
                mlp_dim=config['mlp_dim'], 
                prob_feature=config['prob_feature'], 
                prob_edge=config['prob_edge'],
                tau=config['tau'], 
                alpha=config['alpha'], 
                beta=config['beta'], 
                lambda_cl=config['lambda_cl'], 
                dropout=config['dropout'], 
                lr=config['lr'], 
                seed=cur_seed, 
                epochs=config['epochs'], 
                device=device,
                knn_k=15, 
                phi1=config['phi1'], 
                lambda_byol=config['lambda_byol']
            )
            
            results = test([best_z_test], [best_y_test], 0, target_k, cur_seed)
            print(f"🎯 最终聚类结果: CA={results['CA']:.4f}, NMI={results['NMI']:.4f}, ARI={results['ARI']:.4f}")

if __name__ == "__main__":
    main()