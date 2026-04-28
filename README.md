# scMAGCL: Multi-Granularity Asymmetric Graph Contrastive Learning for Single-Cell and Cross-Omics Representation

![scMAGCL Architecture](./framework.png)

## Introduction

Official PyTorch implementation of **scMAGCL**, a novel framework designed for multi-granularity asymmetric graph contrastive learning to enhance single-cell and cross-omics representation.

## Requirements

The codebase is implemented in Python 3.8+ and PyTorch. To ensure reproducibility, we recommend setting up the environment using the provided `requirements.txt`:

```bash
git clone [https://github.com/Miyao15/scMAGCL.git](https://github.com/Miyao15/scMAGCL.git)
cd scMAGCL
pip install -r requirements.txt
```

Core Dependencies:

* torch : 2.4.1
* torch-geometric : 2.6.1
* scanpy : 1.9.8
* anndata : 0.9.2
* scikit-learn : 1.3.2
* scib : 1.1.5

*(For a complete list of dependencies, please refer to `requirements.txt`.)*

## Datasets

Quick Start Datasets

For convenience, we provide the formatted datasets used in our benchmark experiments. To run the examples in the **Usage** section, please download the following datasets and place them into the `test_data/` directory:

10Xmalt (Cross-Omics scRNA-seq + ADT) : https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

Quake_10x_Limb_Muscle : https://github.com/Philyzh8/scMGCA/tree/master/dataset

Full Raw Datasets

Due to file size limitations, the full raw datasets evaluated in the manuscript are not directly uploaded to this repository. You can download the original data from the following authoritative public repositories or cloud storage. After downloading, please refer to our preprocessing scripts to format the raw data.

Unimodal scRNA-seq Datasets (8 Datasets):

Young: https://zenodo.org/records/7475687

MCA: https://www.dropbox.com/scl/fi/kx9ec2vr922w7aznatc2g/data.zip?rlkey=8xehi0ix40fj965djugny2yq7&e=1

Zeisel: https://www.dropbox.com/scl/fi/kx9ec2vr922w7aznatc2g/data.zip?rlkey=8xehi0ix40fj965djugny2yq7&e=1

Tosches_turtle: https://zenodo.org/records/7475687

Chen: https://zenodo.org/records/7475687

Worm_neuron_cells: https://www.dropbox.com/scl/fi/kx9ec2vr922w7aznatc2g/data.zip?rlkey=8xehi0ix40fj965djugny2yq7&e=1

Quake_10x_Limb_Muscle: https://zenodo.org/records/7475687

Quake_Smart-seq2_Limb_Muscle: https://zenodo.org/records/7475687

Multimodal Benchmark Datasets (8 Datasets):

10Xmalt: https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

10X5kpbmc: https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

GSE150599_spleen_lymph_111: https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

zenodo6348128: https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

human_pbmc_3k: https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

human_pbmc_10x: https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

pbmc_unsorted_10k: https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

GSE201402: https://figshare.com/articles/dataset/scMAGCA-datasets/30164773

Downstream Analysis Datasets (2 Datasets):

human_brain_10x: https://mailustceducn-my.sharepoint.com/personal/hyl2016_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhyl2016%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FMultiomicsBenchmark%2FRaw%5Fdata&viewid=b6d1a33b%2D630a%2D4d98%2Db85e%2D2df575b1c642

GSE163120: https://mailustceducn-my.sharepoint.com/personal/hyl2016_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhyl2016%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FMultiomicsBenchmark%2FRaw%5Fdata&viewid=b6d1a33b%2D630a%2D4d98%2Db85e%2D2df575b1c642

## Usage

Please ensure you have configured the environment properly and placed the downloaded datasets into the `test_data/` directory before execution.

**1. Unimodal scRNA-seq Analysis**

The Young dataset is taken as an example:

```bash
python scMAGCL-main/main.py --data_path "test_data/Young/data.h5" --n_clusters 11 --epochs 200
```

**2. Cross-Omics Integration (RNA + ADT)**

For CITE-seq data, the pipeline performs automated modality alignment and joint representation learning:

```bash
python preprocessing/preprocess_adt.py --rna_h5ad "test_data/4_10Xmalt/10Xmalt_rna.h5ad" --adt_h5ad "test_data/4_10Xmalt/10Xmalt_adt.h5ad" --label_csv "test_data/4_10Xmalt/10Xmalt_label.csv" --filter2 2000 --no_clr --no_scale --train --n_clusters 11
```

**3. Cross-Omics Integration (RNA + ATAC)**

For paired scRNA-seq and scATAC-seq data, use the ATAC preprocessing module. Feature selection (HVGs) is applied independently to both modalities:

```bash
python preprocessing/preprocess_atac.py --atac_h5ad "test_data/20_human_pbmc_3k/human_pbmc_3k_atac.h5ad" --rna_h5ad "test_data/20_human_pbmc_3k/human_pbmc_3k_rna.h5ad" --label_csv "test_data/20_human_pbmc_3k/human_pbmc_3k_label_a.csv" --n_clusters 8 --filter1 --f1 2000 --filter2 --f2 2000 --no_clr --no_scale
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
