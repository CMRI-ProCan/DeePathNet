"""
Script to run the baseline model for the early concatenation methods on the independent test set for drug response prediction.
E.g. python scripts/baseline_independent_test.py configs/sanger_train_ccle_test_gdsc/mutation_cnv_rna_prot/ec_rf_all_genes_mutation_cnv_rna_prot.json
"""
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from tqdm import trange

STAMP = datetime.today().strftime('%Y%m%d%H%M')
OUTPUT_NA_NUM = -100

config_file = sys.argv[1]

# load model configs
configs = json.load(open(config_file, 'r'))

log_suffix = f"{config_file.split('/')[-1].replace('.json', '')}"
if not os.path.isdir(configs['work_dir']):
    os.system(f"mkdir -p {configs['work_dir']}")

log_file = f"{STAMP}_{log_suffix}.log"
logger = logging.getLogger('baseline_ec')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(configs['work_dir'], log_file))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info(open(config_file, 'r').read())
print(open(config_file, 'r').read())

seed = 12345

if configs['task'].lower() == 'classification':
    model_dict = {'rf': RandomForestClassifier(n_jobs=100),
                  'svm': SVC(),
                  'en': ElasticNet(),
                  'svm-linear': SVC(kernel='linear'),
                  'mlp': MLPClassifier(verbose=True)}

else:
    model_dict = {'rf': RandomForestRegressor(n_jobs=100),
                  'svm': SVR(),
                  'en': ElasticNet(),
                  'svm-linear': SVR(kernel='linear'),
                  'mlp': MLPRegressor(verbose=True)}

data_target_train = pd.read_csv(configs['target_file_train'], index_col=0)
data_input_train = pd.read_csv(configs['data_file_train'], index_col=0)
data_target_test = pd.read_csv(configs['target_file_test'], index_col=0)
data_input_test = pd.read_csv(configs['data_file_test'], index_col=0)
data_type = configs['data_type']

common_features = set(data_input_train.columns).intersection(data_input_test.columns)
data_input_train = data_input_train[common_features]
data_input_test = data_input_test[common_features]

genes = np.unique(([x.split("_")[0] for x in data_input_train.columns]))
if 'pathway_file' in configs:
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
    data_input_train = data_input_train[
        [x for x in data_input_train.columns if (x.split("_")[0] in cancer_genes) or (x.split("_")[0] == 'tissue')]]
    data_input_test = data_input_test[
        [x for x in data_input_test.columns if (x.split("_")[0] in cancer_genes) or (x.split("_")[0] == 'tissue')]]

if data_type[0] != 'DR':
    data_input_train = data_input_train[
        [x for x in data_input_train.columns if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)]]
    data_input_test = data_input_test[
        [x for x in data_input_test.columns if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)]]

logger.info(f"Input trainning data shape: {data_input_train.shape}")
logger.info(f"Input trainning target shape: {data_target_train.shape}")
logger.info(f"Input test data shape: {data_input_test.shape}")
logger.info(f"Input test target shape: {data_target_test.shape}")

data_input_train = data_input_train.fillna(0)
data_target_train = data_target_train.fillna(OUTPUT_NA_NUM)
data_input_test = data_input_test.fillna(0)
data_target_test = data_target_test.fillna(OUTPUT_NA_NUM)

train_df = pd.merge(data_target_train, data_input_train, on='Cell_line')
test_df = pd.merge(data_target_test, data_input_test, on='Cell_line')
num_targets = data_target_train.shape[1]

feature_df_list = []
score_df_list = []
params_df_list = []
for i in trange(num_targets):
    y_train = train_df.iloc[:, i]
    X_train = train_df.iloc[:, num_targets:]
    X_train = X_train[(y_train != OUTPUT_NA_NUM)]
    y_train = y_train[y_train != OUTPUT_NA_NUM]

    y_test = test_df.iloc[:, i]
    X_test = test_df.iloc[:, num_targets:]
    X_test = X_test[(y_test != OUTPUT_NA_NUM)]
    y_test = y_test[y_test != OUTPUT_NA_NUM]
    logger.info(f"Running {train_df.columns[i]} Train:{X_train.shape} Test:{X_test.shape}")

    model = model_dict[configs['model']]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    sign = 1 if configs['task'].lower() == 'classification' else -1
    if configs['task'].lower() == 'classification':
        y_confs = model.predict_proba(X_test)
        val_auc = roc_auc_score(y_test, y_confs[:, 1])
        val_acc = accuracy_score(y_test, y_pred)
        score_dict = {'target': train_df.columns[i], 'auc': val_auc,
                      'acc': val_acc}
    else:
        val_mae = mean_absolute_error(y_test, y_pred)
        val_rmse = mean_squared_error(y_test, y_pred, squared=False)
        val_r2 = r2_score(y_test, y_pred)
        val_corr = pearsonr(y_test, y_pred)[0]
        score_dict = {'drug_id': train_df.columns[i],
                      'mae': val_mae, 'rmse': val_rmse,
                      'r2': val_r2, 'corr': val_corr}
    score_df_list.append(score_dict)

logger.info(f"All finished.")
score_df = pd.DataFrame(score_df_list)
logger.info(score_df.median())
if 'save_scores' not in configs or configs['save_scores']:
    score_df.to_csv(f"{configs['work_dir']}/scores_{STAMP}_{log_suffix}.csv", index=False)

