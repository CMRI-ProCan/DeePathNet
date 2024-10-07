import json

from datetime import datetime
import sys
import logging
import os

import pandas as pd

from models import *
from model_transformer_lrp import DOIT_LRP, LRP
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

seed = 12345
torch.manual_seed(seed)
OUTPUT_NA_NUM = -100

config_file = sys.argv[1]
# load model configs
configs = json.load(open(config_file, 'r'))
data_file = configs['data_file']
data_type = configs['data_type']

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = 0
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
data_input_train = data_input[data_input.index.isin(cell_lines_train)]
merged_df_train = pd.merge(data_input_train, data_target_train, on=['Cell_line'])


# test_data = data_input_test
# merged_df_test = pd.merge(test_data, data_target_test, on=['Cell_line'])

def run_lrp(merged_df_train, drug_id=None):
    train_df = merged_df_train.iloc[:, :num_of_features]
    train_target = merged_df_train.iloc[:, num_of_features:]

    X_train = train_df

    train_dataset = MultiOmicDataset(X_train, train_target, mode='train', omics_types=omics_types, logger=None,
                                     with_tissue=with_tissue)

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
    model = DOIT_LRP(len(omics_types), train_target.shape[1], train_dataset.genes_to_id,
                     train_dataset.id_to_genes,
                     pathway_dict, non_cancer_genes, embed_dim=configs['dim'], depth=configs['depth'],
                     num_heads=configs['heads'],
                     mlp_ratio=configs['mlp_ratio'], out_mlp_ratio=configs['out_mlp_ratio'],
                     only_cancer_genes=configs['cancer_only'], tissues=tissues)

    if drug_id:
        drug_file_name = get_model_filename(drug_id)
        drug_path = f"{configs['work_dir']}_{configs['saved_model']}/{configs['saved_model']}_{drug_file_name}.pth"
        if not os.path.exists(drug_path):
            return None
        model.load_state_dict(
            torch.load(drug_path))
    else:
        model.load_state_dict(torch.load(f"{configs['work_dir']}/{configs['saved_model']}"))

    model.cuda()
    model.eval()

    attribution_generator = LRP(model)
    # index = train_target.columns.get_loc('1032;Afatinib;GDSC2')
    pathways = list(pathway_dict.keys())
    if not configs['cancer_only']:
        pathways += ['non_cancer']
    if 'tissue' in data_type:
        pathways += ['tissue']

    res_df_all = []
    for drug_idx in range(train_target.shape[1]):
        transformer_attribution_all = []
        drug = train_target.columns[drug_idx]
        X_train_drug = X_train[X_train.index.isin(train_target[train_target[drug] > 0].index)]
        train_target_drug = train_target[train_target[drug] > 0]
        train_dataset = MultiOmicDataset(X_train_drug, train_target_drug, mode='train', omics_types=omics_types,
                                         logger=None, with_tissue=with_tissue)
        train_loader = DataLoader(train_dataset,
                                  batch_size=1,
                                  num_workers=NUM_WORKERS)
        for i, data in enumerate(train_loader):
            transformer_attribution = attribution_generator.generate_LRP(data,
                                                                         method="transformer_attribution",
                                                                         index=drug_idx).detach().cpu().numpy()
            transformer_attribution_all.append(transformer_attribution[0, :])
        transformer_attribution_sum = np.sum(transformer_attribution_all, axis=0)
        res_df = pd.DataFrame({'pathway': pathways, 'importance': transformer_attribution_sum})
        res_df['drug_id'] = train_target.columns[drug_idx]
        res_df_all.append(res_df)

    res_df_all = pd.concat(res_df_all)
    res_df_all = res_df_all[['drug_id', 'pathway', 'importance']]
    return res_df_all


if 'all_single_mode' in configs and configs['all_single_mode']:
    res_df_all = []
    drug_ids = pd.read_csv(configs['drug_list'], index_col=0).index.values
    for drug_id in tqdm(drug_ids):
        merged_df_train = pd.merge(data_input_train, data_target_train[drug_id], on=['Cell_line'])
        relevance = run_lrp(merged_df_train, drug_id=drug_id)
        if relevance is not None:
            res_df_all.append(relevance)
    res_df_all = pd.concat(res_df_all)
    res_df_all.to_csv(
        f"{configs['work_dir']}_{configs['saved_model']}/explanation_{configs['saved_model']}.csv.gz",
        index=False)
else:
    data_input_train = data_input[data_input.index.isin(cell_lines_train)]
    merged_df_train = pd.merge(data_input_train, data_target_train, on=['Cell_line'])
    res_df_all = run_lrp(merged_df_train)
    res_df_all.to_csv(f"{configs['work_dir']}/explanation_{configs['saved_model'].replace('pth', 'csv')}", index=False)
print("finished")
