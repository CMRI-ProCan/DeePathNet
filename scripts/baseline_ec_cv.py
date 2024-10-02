"""
Script to run the baseline model for the early concatenation methods using cross-validation for drug response prediction.
E.g. python scripts/baseline_ec_cv.py configs/sanger_gdsc_intersection_noprot/mutation_cnv_rna/ec_rf_allgenes_drug_mutation_cnv_rna.json
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from tqdm import trange
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings(action="ignore", category=UserWarning)

STAMP = datetime.today().strftime("%Y%m%d%H%M")
OUTPUT_NA_NUM = -100

config_file = sys.argv[1]

# load model configs
configs = json.load(open(config_file, "r"))

log_suffix = f"{config_file.split('/')[-1].replace('.json', '')}"
if not os.path.isdir(configs["work_dir"]):
    os.system(f"mkdir -p {configs['work_dir']}")

data_file = configs["data_file"]
target_file = configs["target_file"]
data_type = configs["data_type"]

log_file = f"{STAMP}_{log_suffix}.log"
logger = logging.getLogger("baseline_ec")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(configs["work_dir"], log_file))
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info(open(config_file, "r").read())
print(open(config_file, "r").read())

seed = configs["seed"]
cv = KFold(n_splits=configs["cv"], shuffle=True, random_state=seed)

if configs["task"].lower() == "classification":
    model_dict = {
        "lr": LogisticRegression(n_jobs=-1, solver="saga"),
        "rf": RandomForestClassifier(n_jobs=40, max_features="sqrt"),
        "svm": SVC(),
        "en": ElasticNet(),
        "svm-linear": SVC(kernel="linear"),
        "mlp": MLPClassifier(),
        "xgb": XGBClassifier(),
    }

else:
    model_dict = {
        "lr": LinearRegression(),
        "rf": RandomForestRegressor(n_jobs=40, max_features="sqrt"),
        "svm": SVR(),
        "en": ElasticNet(),
        "svm-linear": SVR(kernel="linear"),
        "mlp": MLPRegressor(),
        "xgb": XGBRegressor(),
    }

data_target = pd.read_csv(target_file, index_col=0)

data_input = pd.read_csv(data_file, index_col=0)
genes = np.unique(([x.split("_")[0] for x in data_input.columns]))
if "pathway_file" in configs:
    pathway_dict = {}
    pathway_df = pd.read_csv(configs["pathway_file"])
    if "min_cancer_publication" in configs:
        pathway_df = pathway_df[
            pathway_df["Cancer_Publications"] > configs["min_cancer_publication"]
        ]
        logger.info(
            f"Filtering pathway with Cancer_Publications > {configs['min_cancer_publication']}"
        )
    if "max_gene_num" in configs:
        pathway_df = pathway_df[pathway_df["GeneNumber"] < configs["max_gene_num"]]
        logger.info(f"Filtering pathway with GeneNumber < {configs['max_gene_num']}")
    if "min_gene_num" in configs:
        pathway_df = pathway_df[pathway_df["GeneNumber"] > configs["min_gene_num"]]
        logger.info(f"Filtering pathway with GeneNumber > {configs['min_gene_num']}")

    pathway_df["genes"] = pathway_df["genes"].map(
        lambda x: "|".join([gene for gene in x.split("|") if gene in genes])
    )

    for index, row in pathway_df.iterrows():
        if row["genes"]:
            pathway_dict[row["name"]] = row["genes"].split("|")
    cancer_genes = set([y for x in pathway_df["genes"].values for y in x.split("|")])
    data_input = data_input[
        [
            x
            for x in data_input.columns
            if (x.split("_")[0] in cancer_genes) or (x.split("_")[0] == "tissue")
        ]
    ]

if data_type[0] != "DR":
    data_input = data_input[
        [
            x
            for x in data_input.columns
            if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)
        ]
    ]

logger.info(f"Input data shape: {data_input.shape}")
logger.info(f"Target data shape: {data_target.shape}")

data_input = data_input.fillna(0)
data_target = data_target.fillna(OUTPUT_NA_NUM)

merged_df = pd.merge(data_target, data_input, on="Cell_line")
cell_lines_all = data_input.index.values
num_targets = data_target.shape[1]

count = 0
num_repeat = 1 if "num_repeat" not in configs else configs["num_repeat"]
feature_df_list = []
score_df_list = []
time_df_list = []
logger.info(f"Merged df shape: {merged_df.shape}")

for n in range(num_repeat):
    cv = KFold(n_splits=5, shuffle=True, random_state=(seed + n))
    for cell_lines_train_index, cell_lines_val_index in cv.split(cell_lines_all):
        start_time = time()
        for i in trange(num_targets):
            train_lines = np.array(cell_lines_all)[cell_lines_train_index]
            val_lines = np.array(cell_lines_all)[cell_lines_val_index]
            merged_df_train = merged_df[merged_df.index.isin(train_lines)]
            merged_df_val = merged_df[merged_df.index.isin(val_lines)]

            y_train = merged_df_train.iloc[:, i]
            X_train = merged_df_train.iloc[:, num_targets:]
            X_train = X_train[(y_train != OUTPUT_NA_NUM)]
            y_train = y_train[y_train != OUTPUT_NA_NUM]

            y_val = merged_df_val.iloc[:, i]
            X_val = merged_df_val.iloc[:, num_targets:]
            X_val = X_val[(y_val != OUTPUT_NA_NUM)]
            y_val = y_val[y_val != OUTPUT_NA_NUM]

            model = model_dict[configs["model"]]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            sign = 1 if configs["task"].lower() == "classification" else -1
            seconds_elapsed = time() - start_time
            if configs["task"].lower() == "classification":
                y_confs = model.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, y_confs[:, 1])
                val_acc = accuracy_score(y_val, y_pred)
                score_dict = {
                    "target": merged_df_train.columns[i],
                    "run": f"cv_{count}",
                    "auc": val_auc,
                    "acc": val_acc,
                }
            else:
                val_mae = mean_absolute_error(y_val, y_pred)
                val_rmse = mean_squared_error(y_val, y_pred, squared=False)
                val_r2 = r2_score(y_val, y_pred)
                val_corr = pearsonr(y_val, y_pred)[0]
                score_dict = {
                    "drug_id": merged_df_train.columns[i],
                    "run": f"cv_{count}",
                    "mae": val_mae,
                    "rmse": val_rmse,
                    "r2": val_r2,
                    "corr": val_corr,
                }
            score_df_list.append(score_dict)

            # record feature importance if possible
        end_time = time()
        seconds_elapsed = end_time - start_time
        logger.info(f"cv_{count}: {seconds_elapsed} seconds")
        time_df_list.append({"run": f"cv_{count}", "time": seconds_elapsed})
        count += 1
time_df_list = pd.DataFrame(time_df_list)
logger.info(f"All finished.")
score_df = pd.DataFrame(score_df_list)
# logger.info(score_df.median())
time_df_list.to_csv(f"{configs['work_dir']}/time_{STAMP}_{log_suffix}.csv", index=False)

if "save_scores" not in configs or configs["save_scores"]:
    score_df.to_csv(
        f"{configs['work_dir']}/scores_{STAMP}_{log_suffix}.csv", index=False
    )
# Select only the numeric columns
numeric_columns = score_df.select_dtypes(include=["number"])
# Compute the median of the numeric columns
print(numeric_columns.median())
