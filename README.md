# Transformer-based deep learning integrates multi-omic data with cancer pathways

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Description
--

![Figure1](./figures/Figure1.png)

Transformer-based deep learning integrates multi-omic data with cancer pathways.
Cai, et al., 2023

# Pre-requisite

1. Follow https://docs.anaconda.com/free/anaconda/install/index.html to set up the Python environment with Anaconda
2. Follow https://pytorch.org/ to install PyTorch


# Usage

To run the examples below, please: 
1. Download the relevant files from https://doi.org/10.6084/m9.figshare.24137619
2. Update the file paths specified in each config file according to the downloaded location in step 1. The file path is currently set with an absolute path, starting with `/home/scai/DeePathNet/`, which should be replaced by the actual file locations.

## Drug response prediction:

```python scripts/deepathnet_independent_test.py configs/sanger_train_ccle_test_gdsc/mutation_cnv_rna_prot/deepathnet_mutation_cnv_rna_prot_random_control.json```

## TCGA cancer type classification

```python scripts/deepathnet_cv.py configs/tcga_all_cancer_types/mutation_cnv_rna/deepathnet_mutation_cnv_rna.json```

## Breast cancer subtype classification

```python scripts/deepathnet_independent_test.py configs/tcga_train_cptac_test_brca/cnv_rna/deepathnet_cnv_rna.json```

## Raising issues
Please kindly note that issues may occur due to various differences in computational environments. We recommend using 
the same versions of Python and PyTorch as we used in our study, described in the Methods sections of the manuscript. 
If you have any issue running the tool, please raise a ticket using the Issue tab and we will respond and help as soon as possible.
**We endeavour to provide support and improve DeePathNet as much as we can.**

Contact
--
For more information, please contact the study authors. Contact details are available in the associated publication. 
