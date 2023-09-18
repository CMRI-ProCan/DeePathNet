# Transformer-based deep learning integrates multi-omic data with cancer pathways

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Description
--

![Figure1](./figures/Figure1.png)

Transformer-based deep learning integrates multi-omic data with cancer pathways.
Cai, et al., 2022

# Usage
To run the examples below, please also download the relevant files from https://doi.org/10.6084/m9.figshare.24137619

## Drug response prediction: 
```python scripts/deepathnet_independent_test.py configs/sanger_train_ccle_test_gdsc/mutation_cnv_rna_prot/deepathnet_mutation_cnv_rna_prot_random_control.json```
## TCGA cancer type classification
```python scripts/deepathnet_cv.py configs/tcga_all_cancer_types/mutation_cnv_rna/deepathnet_mutation_cnv_rna.json```
## Breast cancer subtype classification
```python scripts/deepathnet_independent_test.py configs/tcga_train_cptac_test_brca/cnv_rna/deepathnet_cnv_rna.json```

Contact
--
For more information, please contact the study authors. Contact details are available in the associated publication. 
