# scMAGCL: Multi-Granularity Asymmetric Graph Contrastive Learning for Single-Cell and Cross-Omics Representation

![scMAGCL Architecture](./framework.png)

## Requirements

* python : 3.8.17
* scanpy : 1.9.6
* sklearn : 1.2.2
* torch : 1.8.1
* torch-geometric : 2.2.0

## Datasets

**Unimodal scRNA-seq Datasets:**
* Tosches_turtle : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110204
* Chen : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87544
* Zeisel : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361
* Quake_10x_Limb_Muscle : https://figshare.com/articles/dataset/Tabula_Muris/5829687
* human_pbmc_3k : https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k

**Multimodal Datasets (scRNA-seq + ADT / scATAC-seq):**
* GSE150599_spleen_lymph_111 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150599
* GSE201402 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201402
* GSE163120 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163120
* human_brain_10x : https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets

## Usage

Please navigate to the directory containing the core scripts (`main.py`, `utils.py`, `config.py`, `scMAGCL.py`) before execution. The `Quake_10x_Limb_Muscle` dataset is taken as an example:

```bash
cd scMAGCL-main
python main.py --data_path '../data/Quake_10x_Limb_Muscle.h5' --save_model_path '../save_file' --n_clusters 6