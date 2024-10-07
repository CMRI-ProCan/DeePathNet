import json

from datetime import datetime
import sys
import logging
import os
import sys

import pandas as pd

sys.path.append(os.getcwd() + '/..')
from models import *
from model_transformer_lrp import DOIT_LRP, LRP
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import shap
import time

seed = 12345
torch.manual_seed(seed)
OUTPUT_NA_NUM = -100

config_file = sys.argv[1]

mode = 'grad' if len(sys.argv) <= 2 else sys.argv[2]
print(f"SHAP algo: {mode}")
# load model configs
configs = json.load(open(config_file, 'r'))
data_file = configs['data_file']
data_type = configs['data_type']

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = 0
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'

data_target = pd.read_csv(configs['target_file'], low_memory=False, index_col=0)
drug_ids = data_target.columns
if 'drug_id' in configs and configs['drug_id'] != "":
    data_target = data_target[configs['drug_id']]
    drug_ids = [configs['drug_id']]

data_input = pd.read_csv(data_file, index_col=0)
if data_type[0] != 'DR':
    data_input = data_input[
        [x for x in data_input.columns if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)]]

tissues = [x for x in data_input.columns if 'tissue' in x] if 'tissue' in data_type else None
omics_types = [x for x in data_type if x != 'tissue']
with_tissue = 'tissue' in data_type

with open(configs['train_cells']) as f:
    cell_lines_train = [line.rstrip() for line in f]
with open(configs['test_cells']) as f:
    cell_lines_test = [line.rstrip() for line in f]

genes = np.unique(([x.split("_")[0] for x in data_input.columns if x.split("_")[0] != 'tissue']))

data_target_train = data_target[
    data_target.index.isin(cell_lines_train)]
data_target_test = data_target[
    data_target.index.isin(cell_lines_test)]

num_of_features = data_input.shape[1]

pathway_dict = {}
pathway_df = pd.read_csv(configs['pathway_file'])

pathway_df['genes'] = pathway_df['genes'].map(
    lambda x: "|".join([gene for gene in x.split('|') if gene in genes]))
if 'min_cancer_publication' in configs:
    pathway_df = pathway_df[pathway_df['Cancer_Publications'] > configs['min_cancer_publication']]
if 'max_gene_num' in configs:
    pathway_df = pathway_df[pathway_df['GeneNumber'] < configs['max_gene_num']]
if 'min_gene_num' in configs:
    pathway_df = pathway_df[pathway_df['GeneNumber'] > configs['min_gene_num']]

for index, row in pathway_df.iterrows():
    pathway_dict[row['name']] = row['genes'].split('|')

cancer_genes = set([y for x in pathway_df['genes'].values for y in x.split("|")])
non_cancer_genes = sorted(set(genes) - set(cancer_genes))

data_input_train = data_input[data_input.index.isin(cell_lines_train)]
data_input_test = data_input[data_input.index.isin(cell_lines_test)]


def run_shap(merged_df_train, merged_df_test, drug_ids=None):
    train_df = merged_df_train.iloc[:, :num_of_features]
    test_df = merged_df_test.iloc[:, :num_of_features]
    train_target = merged_df_train.iloc[:, num_of_features:]
    test_target = merged_df_test.iloc[:, num_of_features:]

    X_train = train_df
    X_test = test_df

    train_dataset = MultiOmicDataset(X_train, train_target, mode='train', omics_types=omics_types, logger=None,
                                     with_tissue=with_tissue)
    test_dataset = MultiOmicDataset(X_test, test_target, mode='val', omics_types=omics_types, logger=None,
                                    with_tissue=with_tissue)

    model = DOIT_LRP(len(omics_types), train_target.shape[1], train_dataset.genes_to_id,
                     train_dataset.id_to_genes,
                     pathway_dict, non_cancer_genes, embed_dim=configs['dim'], depth=configs['depth'],
                     num_heads=configs['heads'],
                     mlp_ratio=configs['mlp_ratio'], out_mlp_ratio=configs['out_mlp_ratio'],
                     only_cancer_genes=configs['cancer_only'], tissues=tissues)
    if len(drug_ids) == 1:
        drug_file_name = get_model_filename(drug_ids[0])
        drug_path = f"{configs['work_dir']}_{configs['saved_model']}/{configs['saved_model']}_{drug_file_name}.pth"
        if not os.path.exists(drug_path):
            return None, None
        model.load_state_dict(torch.load(drug_path))
    else:
        model.load_state_dict(torch.load(f"{configs['work_dir']}/{configs['saved_model']}"))
    model.to(device)
    model.eval()

    train_loader = DataLoader(train_dataset,
                              batch_size=len(train_dataset), shuffle=True,
                              num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset,
                             batch_size=len(test_dataset), shuffle=True,
                             num_workers=NUM_WORKERS)

    data = next(iter(train_loader))
    test_data = next(iter(test_loader))

    tissue_x = None
    test_tissue_x = None
    if len(data) == 2:
        (input, targets) = data
        (test_input, test_targets) = test_data
    elif len(data) == 3:
        (input, tissue_x, targets) = data
        (test_input, test_tissue_x, test_targets) = test_data
    else:
        raise Exception

    NUM_EXPLAINED = 400
    N_SAMPLES = 50

    start = time.time()
    if mode == 'grad':
        if tissue_x is not None:
            background = [input.float().to(device), tissue_x.float().to(device)]
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(
                [test_input[:NUM_EXPLAINED, :, :].float().to(device),
                 test_tissue_x[:NUM_EXPLAINED, :].float().to(device)],
                nsamples=N_SAMPLES)
        else:
            background = input.float().to(device)
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(test_input[:NUM_EXPLAINED, :, :].float().to(device), nsamples=N_SAMPLES)
    else:
        raise Exception
    end = time.time()
    print(end - start)

    all_drug_gradients_summary = {'drug_id': [], 'gene': []}
    all_drug_gradients_tissue_summary = {'drug_id': [], 'tissue': [], 'importance': []}
    for target in omics_types:
        all_drug_gradients_summary[target] = []

    for drug_idx in range(len(drug_ids)):
        drug_id = drug_ids[drug_idx]
        if len(drug_ids) > 1:
            omics_shap = shap_values[drug_idx][0]  # N x genes x num_omics
            tissue_shap = shap_values[drug_idx][1]  # N x tissue
        else:
            omics_shap = shap_values[0]  # N x genes x num_omics
            tissue_shap = shap_values[1]  # N x tissue

        omics_shap_mean = np.mean(np.abs(omics_shap), axis=0)
        tissue_shap_mean = np.mean(np.abs(tissue_shap), axis=0)

        all_drug_gradients_summary['drug_id'].extend([drug_id] * len(genes))
        all_drug_gradients_summary['gene'].extend(genes)
        for i in range(len(omics_types)):
            all_drug_gradients_summary[omics_types[i]].extend(omics_shap_mean[:, i])

        all_drug_gradients_tissue_summary['drug_id'].extend([drug_id] * len(tissues))
        all_drug_gradients_tissue_summary['tissue'].extend(tissues)
        all_drug_gradients_tissue_summary['importance'].extend(tissue_shap_mean)

    all_drug_gradients_summary = pd.DataFrame(all_drug_gradients_summary)
    all_drug_gradients_summary['sum'] = all_drug_gradients_summary.iloc[:, 2:].sum(axis=1)

    all_drug_gradients_tissue_summary = pd.DataFrame(all_drug_gradients_tissue_summary)

    del model, explainer, input, targets
    torch.cuda.empty_cache()
    return all_drug_gradients_summary, all_drug_gradients_tissue_summary


if 'all_single_mode' in configs and configs['all_single_mode']:
    all_drug_gradients_summary = []
    all_drug_gradients_tissue_summary = []
    drug_ids = pd.read_csv(configs['drug_list'], index_col=0).index.values
    for drug_id in tqdm(drug_ids):
        merged_df_train = pd.merge(data_input_train, data_target_train[drug_id], on=['Cell_line'])
        merged_df_test = pd.merge(data_input_test, data_target_test[drug_id], on=['Cell_line'])
        omics, tissue = run_shap(merged_df_train, merged_df_test, drug_ids=[drug_id])
        if omics is not None:
            all_drug_gradients_summary.append(omics)
            all_drug_gradients_tissue_summary.append(tissue)

    all_drug_gradients_summary = pd.concat(all_drug_gradients_summary)
    all_drug_gradients_tissue_summary = pd.concat(all_drug_gradients_tissue_summary)
    all_drug_gradients_summary.to_csv(
        f"{configs['work_dir']}_{configs['saved_model']}/shap{mode}_genes_{configs['saved_model']}.csv.gz",
        index=False)
    all_drug_gradients_tissue_summary.to_csv(
        f"{configs['work_dir']}_{configs['saved_model']}/shap{mode}_tissue_{configs['saved_model']}.csv.gz",
        index=False)
else:
    merged_df_train = pd.merge(data_input_train, data_target_train, on=['Cell_line'])
    merged_df_test = pd.merge(data_input_test, data_target_test, on=['Cell_line'])
    all_drug_gradients_summary, all_drug_gradients_tissue_summary = run_shap(merged_df_train, merged_df_test,
                                                                             drug_ids=drug_ids)
    all_drug_gradients_summary.to_csv(
        f"{configs['work_dir']}/shap{mode}_genes_{configs['saved_model'].replace('pth', 'csv.gz')}",
        index=False)
    all_drug_gradients_tissue_summary.to_csv(
        f"{configs['work_dir']}/shap{mode}_tissue_{configs['saved_model'].replace('pth', 'csv.gz')}", index=False)
