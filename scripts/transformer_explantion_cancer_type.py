"""
Script to run pathway-level model explanation for cancer type
E.g. python scripts/transformer_explantion_cancer_type.py configs/tcga_brca_subtypes/mutation_cnv_rna/deepathnet_allgenes_mutation_cnv_rna.json
"""
import json
import os
import sys

from sklearn.model_selection import KFold

sys.path.append(os.getcwd() + '/..')
from models import *
from model_transformer_lrp import DeePathNet, LRP
from torch.utils.data import DataLoader
from tqdm import trange

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
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'

data_target = pd.read_csv(configs['target_file'], low_memory=False, index_col=0)

data_input = pd.read_csv(data_file, index_col=0)
if data_type[0] != 'DR':
    data_input = data_input[
        [x for x in data_input.columns if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)]]

omics_types = [x for x in data_type if x != 'tissue']

genes = np.unique(([x.split("_")[0] for x in data_input.columns if x.split("_")[0] != 'tissue']))
class_name_to_id = dict(
    zip(sorted(data_target.iloc[:, 0].unique()), list(range(data_target.iloc[:, 0].unique().size))))
id_to_class_name = dict(
    zip(list(range(data_target.iloc[:, 0].unique().size)), sorted(data_target.iloc[:, 0].unique())))


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

cell_lines_all = data_input.index.values
cv = KFold(n_splits=5, shuffle=True, random_state=seed)
cell_lines_train_index, cell_lines_val_index = next(cv.split(cell_lines_all))
cell_lines_train = np.array(cell_lines_all)[cell_lines_train_index]
cell_lines_test = np.array(cell_lines_all)[cell_lines_val_index]

data_input_train = data_input[data_input.index.isin(cell_lines_train)]
data_input_test = data_input[data_input.index.isin(cell_lines_test)]
data_target_train = data_target[
    data_target.index.isin(cell_lines_train)]
data_target_test = data_target[
    data_target.index.isin(cell_lines_test)]

def run_lrp_cancer_type(merged_df_train):
    train_df = merged_df_train.iloc[:, :num_of_features]
    train_target = merged_df_train.iloc[:, num_of_features:]

    X_train = train_df

    train_dataset = MultiOmicMulticlassDataset(X_train, train_target, mode='train', omics_types=omics_types,
                                               class_name_to_id=class_name_to_id, logger=None)

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
    model = DeePathNet(len(omics_types), len(class_name_to_id), train_dataset.genes_to_id,
                     train_dataset.id_to_genes,
                     pathway_dict, non_cancer_genes, embed_dim=configs['dim'], depth=configs['depth'],
                     num_heads=configs['heads'],
                     mlp_ratio=configs['mlp_ratio'], out_mlp_ratio=configs['out_mlp_ratio'],
                     only_cancer_genes=configs['cancer_only'])
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
    for idx in trange(len(class_name_to_id)):
        transformer_attribution_all = []
        cancer_type = id_to_class_name[idx]

        train_loader = DataLoader(train_dataset,
                                  batch_size=1,
                                  num_workers=NUM_WORKERS)
        for i, data in enumerate(train_loader):
            transformer_attribution = attribution_generator.generate_LRP(data,
                                                                         method="transformer_attribution",
                                                                         index=idx).detach().cpu().numpy()
            transformer_attribution_all.append(transformer_attribution[0, :])
        transformer_attribution_sum = np.sum(transformer_attribution_all, axis=0)
        res_df = pd.DataFrame({'pathway': pathways, 'importance': transformer_attribution_sum})
        res_df['cancer_type'] = cancer_type
        res_df_all.append(res_df)

    res_df_all = pd.concat(res_df_all)
    res_df_all = res_df_all[['cancer_type', 'pathway', 'importance']]
    return res_df_all


merged_df_train = pd.merge(data_input_train, data_target_train, on=['Cell_line'])
res_df_all = run_lrp_cancer_type(merged_df_train)
res_df_all.to_csv(f"{configs['work_dir']}/explanation_{configs['saved_model'].replace('pth', 'csv')}", index=False)
print("finished")
