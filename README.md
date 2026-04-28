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

The preprocessed datasets used for benchmarking in this study are publicly available for reproducibility:

* 10Xmalt (Cross-Omics scRNA-seq + ADT) : [https://figshare.com/articles/dataset/scMAGCA-datasets/30164773](https://figshare.com/articles/dataset/scMAGCA-datasets/30164773)
* Unimodal scRNA-seq Benchmark Datasets : [https://github.com/Philyzh8/scMGCA/tree/master/dataset](https://github.com/Philyzh8/scMGCA/tree/master/dataset)

### Full Raw Datasets (Data Availability)

Due to file size limitations, the full raw datasets evaluated in the manuscript (16 datasets in total) are not directly uploaded to this repository. You can download the original data from the following authoritative public repositories. After downloading, please refer to our preprocessing scripts (e.g., `preprocess_adt.py`, `preprocess_atac.py`) to format the raw data.

**Unimodal scRNA-seq Datasets (Table 1):**
* **Tosches_turtle**: [https://hemberg-lab.github.io/scRNA.seq.datasets/animal/brain/](https://hemberg-lab.github.io/scRNA.seq.datasets/animal/brain/)
* **Chen**: [https://hemberg-lab.github.io/scRNA.seq.datasets/human/brain/](https://hemberg-lab.github.io/scRNA.seq.datasets/human/brain/)
* **Zeisel**: [https://hemberg-lab.github.io/scRNA.seq.datasets/mouse/brain/](https://hemberg-lab.github.io/scRNA.seq.datasets/mouse/brain/)
* **Young**: [https://hemberg-lab.github.io/scRNA.seq.datasets/human/kidney/](https://hemberg-lab.github.io/scRNA.seq.datasets/human/kidney/)
* **Worm_neuron_cells**: [https://hemberg-lab.github.io/scRNA.seq.datasets/animal/worm/](https://hemberg-lab.github.io/scRNA.seq.datasets/animal/worm/)
* **Quake_10x_Limb_Muscle**: [https://figshare.com/projects/Tabula_Muris/27733](https://figshare.com/projects/Tabula_Muris/27733)
* **Quake_Smart-seq2_Limb_Muscle**: [https://figshare.com/projects/Tabula_Muris/27733](https://figshare.com/projects/Tabula_Muris/27733)
* **human_pbmc_3k**: [https://www.10xgenomics.com/resources/datasets](https://www.10xgenomics.com/resources/datasets)

**Multimodal Cross-Omics Datasets (Table 2):**
* **GSE150599_spleen_lymph_111**: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150599](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150599)
* **GSE163120** (Glioblastoma): [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163120](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163120)
* **GSE201402** (Gastric Cancer): [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201402](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201402)
* **SHARE-seq_mouse_skin**: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203)
* **10X5kpbmc**: [https://www.10xgenomics.com/resources/datasets](https://www.10xgenomics.com/resources/datasets)
* **human_pbmc_3k**: [https://www.10xgenomics.com/resources/datasets](https://www.10xgenomics.com/resources/datasets)
* **human_pbmc_10x**: [https://www.10xgenomics.com/resources/datasets](https://www.10xgenomics.com/resources/datasets)
* **human_brain_10x**: [https://www.10xgenomics.com/resources/datasets](https://www.10xgenomics.com/resources/datasets)

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
