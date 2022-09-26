import torch.optim
from sklearn.model_selection import KFold
import logging
import sys
import json
import os
from datetime import datetime

from tqdm import tqdm

from optimizer.radam import RAdam
from optimizer.adamw import AdamW
from torch.utils.data import DataLoader
from utils.training_prepare import prepare_data

from models import *
from model_transformer_lrp import DOIT_LRP

STAMP = datetime.today().strftime('%Y%m%d%H%M')

config_file = sys.argv[1]
# load model configs
configs = json.load(open(config_file, 'r'))

log_suffix = ''
if 'suffix' in configs:
    log_suffix = configs['suffix']

seed = 12345
torch.manual_seed(seed)
cv = KFold(n_splits=5, shuffle=True, random_state=seed)

BATCH_SIZE = configs['batch_size']
NUM_WORKERS = 0
LOG_FREQ = configs['log_freq']
NUM_EPOCHS = configs['num_of_epochs']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_setup(genes_to_id, id_to_genes, target_dim):
    def load_pathway():
        pathway_dict = {}
        pathway_df = pd.read_csv(configs['pathway_file'])
        if 'min_cancer_publication' in configs:
            pathway_df = pathway_df[pathway_df['Cancer_Publications'] > configs['min_cancer_publication']]
            logger.info(f"Filtering pathway with Cancer_Publications > {configs['min_cancer_publication']}")
        if 'max_gene_num' in configs:
            pathway_df = pathway_df[pathway_df['GeneNumber'] < configs['max_gene_num']]
            logger.info(f"Filtering pathway with GeneNumber < {configs['max_gene_num']}")
        if 'min_gene_num' in configs:
            pathway_df = pathway_df[pathway_df['GeneNumber'] > configs['min_gene_num']]
            logger.info(f"Filtering pathway with GeneNumber > {configs['min_gene_num']}")

        pathway_df['genes'] = pathway_df['genes'].map(
            lambda x: "|".join([gene for gene in x.split('|') if gene in genes]))

        for index, row in pathway_df.iterrows():
            pathway_dict[row['name']] = row['genes'].split('|')
        cancer_genes = set([y for x in pathway_df['genes'].values for y in x.split("|")])
        non_cancer_genes = set(genes) - set(cancer_genes)
        logger.info(f"Cancer genes:{len(cancer_genes)}\tNon-cancer genes:{len(non_cancer_genes)}")
        return pathway_dict, non_cancer_genes

    if configs['model'] == 'DeepMultiOmicNetV3':
        model = DeepMultiOmicNetV3(len(genes), len(omics_types), target_dim,
                                   configs['hidden_width'], configs['hidden_size'], group=configs['group'])
    elif configs['model'] == 'DeepMultiOmicNetV3S':
        model = DeepMultiOmicNetV3S(len(genes), len(omics_types), target_dim,
                                    configs['hidden_width'], configs['hidden_size'])
    elif configs['model'] == 'DeepMultiOmicPathwayNet':
        pathway_dict, non_cancer_genes = load_pathway()
        model = DeepMultiOmicPathwayNet(configs['hidden_width'], len(omics_types), target_dim, genes_to_id,
                                        id_to_genes,
                                        pathway_dict, non_cancer_genes, equal_width=configs['equal_width'],
                                        only_cancer_genes=configs['cancer_only'])
    elif configs['model'] == 'DOIT_LRP':
        pathway_dict, non_cancer_genes = load_pathway()
        model = DOIT_LRP(len(omics_types), target_dim, genes_to_id,
                         id_to_genes,
                         pathway_dict, non_cancer_genes, embed_dim=configs['dim'], depth=configs['depth'],
                         mlp_ratio=configs['mlp_ratio'], out_mlp_ratio=configs['out_mlp_ratio'],
                         num_heads=configs['heads'], pathway_drop_rate=configs['pathway_dropout'],
                         only_cancer_genes=configs['cancer_only'], tissues=tissues)
        logger.info(open("/home/scai/DeePathNet/scripts/model_transformer_lrp.py", 'r').read())
    else:
        raise Exception

    logger.info(model)
    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])

    logger.info(optimizer)

    lr_scheduler = None

    return model, criterion, optimizer, lr_scheduler


def run_experiment(merged_df_train, merged_df_test, val_score_dict, run='test', class_name_to_id=None):
    train_df = merged_df_train.iloc[:, :num_of_features]
    test_df = merged_df_test.iloc[:, :num_of_features]
    train_target = merged_df_train.iloc[:, num_of_features:]
    test_target = merged_df_test.iloc[:, num_of_features:]

    X_train = train_df
    X_test = test_df

    if configs['task'] == 'multiclass':
        train_dataset = MultiOmicMulticlassDataset(X_train, train_target, mode='train', omics_types=omics_types,
                                                   class_name_to_id=class_name_to_id, logger=logger)
        test_dataset = MultiOmicMulticlassDataset(X_test, test_target, mode='val', omics_types=omics_types,
                                                  class_name_to_id=class_name_to_id, logger=logger)
    else:
        train_dataset = MultiOmicDataset(X_train, train_target, mode='train', omics_types=omics_types, logger=logger,
                                         with_tissue=with_tissue)
        test_dataset = MultiOmicDataset(X_test, test_target, mode='val', omics_types=omics_types, logger=logger,
                                        with_tissue=with_tissue)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=configs['drop_last'],
                              num_workers=NUM_WORKERS)

    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

    if configs['task'] == 'multiclass':
        target_dim = len(class_name_to_id)
    else:
        target_dim = train_target.shape[1]
    model, criterion, optimizer, lr_scheduler = get_setup(train_dataset.genes_to_id, train_dataset.id_to_genes,
                                                          target_dim)

    val_drug_ids = merged_df_test.columns[num_of_features:]
    train_res, test_res = train_loop(NUM_EPOCHS, train_loader, test_loader, model, criterion, optimizer, logger, STAMP,
                                     configs, lr_scheduler, val_drug_ids, run=run, val_score_dict=val_score_dict)
    return train_res, test_res


data_dict = prepare_data(config_file, STAMP)
data_input_train = data_dict['data_input_train']
data_input_test = data_dict['data_input_test']
data_target_train = data_dict['data_target_train']
data_target_test = data_dict['data_target_test']
val_score_dict = data_dict['val_score_dict']
num_of_features = data_dict['num_of_features']
genes = data_dict['genes']
omics_types = data_dict['omics_types']
with_tissue = data_dict['with_tissue']
tissues = data_dict['tissues']
cell_lines_train = data_dict['cell_lines_train']
logger = data_dict['logger']

class_name_to_id = None
if configs['task'] == 'multiclass':
    class_name_to_id = dict(
        zip(sorted(data_target_train.iloc[:, 0].unique()), list(range(data_target_train.iloc[:, 0].unique().size))))

if configs['do_cv']:
    count = 0
    for cell_lines_train_index, cell_lines_val_index in cv.split(cell_lines_train):
        train_lines = np.array(cell_lines_train)[cell_lines_train_index]
        val_lines = np.array(cell_lines_train)[cell_lines_val_index]

        merged_df_train = pd.merge(data_input_train[data_input_train.index.isin(train_lines)],
                                   data_target_train, on=['Cell_line'])

        val_data = data_input_train[data_input_train.index.isin(val_lines)]

        merged_df_val = pd.merge(val_data,
                                 data_target_train,
                                 on=['Cell_line'])

        train_res, val_res = run_experiment(merged_df_train, merged_df_val, val_score_dict, run=f"cv_{count}",
                                            class_name_to_id=class_name_to_id)
        count += 1

logger.info("CV finished. Now running full training")
test_data = data_input_test

if 'all_single_mode' in configs and configs['all_single_mode']:
    assert not (configs['all_single_mode'] and configs['do_cv'])  # we do not tune models for single drug mode
    if 'drug_list' in configs and configs['drug_list']:
        drug_ids = pd.read_csv(configs['drug_list'], index_col=0).index.values
    else:
        drug_ids = data_target_train.columns

    for drug_id in tqdm(drug_ids):
        logger.info(f"Now Running drug: {drug_id}")
        merged_df_train = pd.merge(data_input_train, data_target_train[drug_id], on=['Cell_line'])
        merged_df_test = pd.merge(test_data, data_target_test[drug_id], on=['Cell_line'])
        run_experiment(merged_df_train, merged_df_test, val_score_dict, run='test', class_name_to_id=class_name_to_id)
else:
    merged_df_train = pd.merge(data_input_train, data_target_train, on=['Cell_line'])
    merged_df_test = pd.merge(test_data, data_target_test, on=['Cell_line'])
    run_experiment(merged_df_train, merged_df_test, val_score_dict, run='test', class_name_to_id=class_name_to_id)

if 'save_scores' not in configs or configs['save_scores']:
    val_score_df = pd.DataFrame(val_score_dict)
    val_score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv.gz", index=False)
logger.info("Full training finished.")
