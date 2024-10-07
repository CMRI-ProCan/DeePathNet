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

## Data input
The input files are specified in the config files, such as ```configs/sanger_train_ccle_test_gdsc/mutation_cnv_rna_prot/deepathnet_mutation_cnv_rna_prot_random_control.json```.
The input data file should have the *samples as rows* and *features as columns*. Each feature should contain an `_`, which separates the gene name and the omic data type. The data type needs to match the config files. Below is a mocked example of the data input.
| Sample     | GeneA_RNA | GeneA_PROT | GeneB_RNA | GeneB_PROT |
|------------|-----------|------------|-----------|------------|
| Cell_lineA | 10        | 8          | 2         | 3          |
| Cell_lineB | 15        | 12         | 1         | 2          |
| Cell_lineC | 5         | 3          | 10        | 8          |

Real data files can be found at the figshare repo https://doi.org/10.6084/m9.figshare.24137619 

## Data output
The output of DeePathNet contains the predictions for drug response (IC50) or cancer types/subtypes.

## Drug response prediction:

```python scripts/deepathnet_independent_test.py configs/sanger_train_ccle_test_gdsc/mutation_cnv_rna_prot/deepathnet_mutation_cnv_rna_prot_random_control.json```

## TCGA cancer type classification

```python scripts/deepathnet_cv.py configs/tcga_all_cancer_types/mutation_cnv_rna/deepathnet_mutation_cnv_rna.json```

## Breast cancer subtype classification

```python scripts/deepathnet_independent_test.py configs/tcga_train_cptac_test_brca/cnv_rna/deepathnet_cnv_rna.json```

## Running moCluster and mixOmics for comparison

Scripts for running moCluster and mixOmics are also provided for comparison with DeePathNet. They can be found in the `R` directory.

Then, run the following command to compare the results of DeePathNet with moCluster after the dimensionality reduction:
```python scripts/baseline_ec_cv.py configs/sanger_gdsc_intersection_noprot/mutation_cnv_rna/moCluster_rf_allgenes_drug_mutation_cnv_rna.json```
``` python scripts/cancer_type_baseline_23cancertypes.py```

## Running feature importance analysis
To calculate pathway-level feature importance, run the following command:
```python scripts/transformer_explantion_cancer_type.py configs/tcga_brca_subtypes/mutation_cnv_rna/deepathnet_allgenes_mutation_cnv_rna.json```
To calculate gene-level feature importance, run the following command:
```python scripts/transformer_shap_cancer_type.py configs/tcga_brca_subtypes/mutation_cnv_rna/deepathnet_allgenes_mutation_cnv_rna.json```

### Calculating feature importance for other tasks (not analysed in the study)
To calculate feature importance for other tasks, such as drug response prediction, please update the config file accordingly. For example:
```python scripts/transformer_explantion_drug_response.py configs/sanger_train_ccle_test_gdsc/mutation_cnv_rna_prot/deepathnet_mutation_cnv_rna.json```
```python scripts/transformer_shap_drug_response.py configs/sanger_train_ccle_test_gdsc/mutation_cnv_rna_prot/deepathnet_mutation_cnv_rna_prot.json```

## Raising issues
Please kindly note that issues may occur due to various differences in computational environments. We recommend using 
the same versions of Python and PyTorch as we used in our study, described in the Methods sections of the manuscript. 
If you have any issue running the tool, please raise a ticket using the Issue tab and we will respond and help as soon as possible.
**We endeavour to provide support and improve DeePathNet as much as we can.**

Contact
--
For more information, please contact the study authors. Contact details are available in the associated publication. 
