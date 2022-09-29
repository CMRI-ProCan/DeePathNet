"""
Script to run baseline models for breast cancer subtype classification using independent test set
E.g. python cancer_type_baseline_brca_validation.py
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

NUM_REPATE = 5
num_fold = 5

seed = 1

def run_model(input_df_train, input_df_test, clf_name, data_type=('cnv', 'rna')):
    count = 0
    clf_results_df = []
    if data_type[0] != 'DR':
        input_df_train = input_df_train[[
            x for x in input_df_train.columns
            if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)
        ]]
    num_of_features = input_df_train.shape[1]

    if clf_name == "RF":
        clf = RandomForestClassifier(n_jobs=40, max_features='sqrt')
    elif clf_name == "XGB":
        clf = XGBClassifier(n_jobs=60)
    elif clf_name == "LR":
        clf = LogisticRegression(n_jobs=40, solver='saga')
    else:
        raise Exception

    merged_df_train = pd.merge(
        input_df_train,
        target_df_train,
        on=['Cell_line'])

    merged_df_val = pd.merge(input_df_test, target_df_test, on=['Cell_line'])
    clf.fit(merged_df_train.iloc[:, :-1],
            merged_df_train.iloc[:, -1].values.flatten())
    y_pred = clf.predict(merged_df_val.iloc[:, :-1])
    y_conf = clf.predict_proba(merged_df_val.iloc[:, :-1])
    y_true = merged_df_val.iloc[:, -1].values.flatten()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_conf, multi_class='ovo')
    clf_results_df.append({
        'run': f'cv_{count}',
        'acc': acc,
        'f1': f1,
        'roc_auc': auc
    })
    val_res_perclass = {}
    val_res_perclass['y_pred'] = y_pred
    val_res_perclass['y_true'] = y_true
    for i in range(y_conf.shape[1]):
        val_res_perclass[f"feature_{i}"] = y_conf[:, i]

    clf_results_df = pd.DataFrame(clf_results_df)
    return clf_results_df, pd.DataFrame(val_res_perclass)


input_df_train = pd.read_csv(
    "../data/processed/omics/tcga_brca_as_validation.csv",
    index_col=0)
target_df_train = pd.read_csv(
    "../data/processed/cancer_type/tcga_brca_mutation_cnv_rna_subtypes.csv",
    index_col=0)
input_df_test = pd.read_csv(
    "../data/processed/omics/cptac_as_validation.csv",
    index_col=0)
target_df_test = pd.read_csv(
    "../data/processed/cancer_type/cptac_brca_cnv_rna_subtypes_independent.csv",
    index_col=0)
# target_df_train = pd.read_csv(
#     "../data/processed/cancer_type/tcga_23_cancer_types_mutation_cnv_rna.csv",
#     index_col=0)

class_name_to_id = dict(
    zip(sorted(target_df_train.iloc[:, 0].unique()),
        list(range(target_df_train.iloc[:, 0].unique().size))))
id_to_class_name = dict(
    zip(list(range(target_df_train.iloc[:, 0].unique().size)), sorted(target_df_train.iloc[:, 0].unique())))

input_df_train = input_df_train.fillna(0)
input_df_test = input_df_test.fillna(0)
# %% BRCA
print("Running LR")
lr_results_df, all_val_df = run_model(input_df_train, input_df_test, "LR")
all_val_df.columns = [id_to_class_name[int(x.split("_")[-1])] if "feature_" in x else x for x in
                          all_val_df.columns]
lr_results_df.to_csv("../results/tcga_brca_subtype/lr_cnv_rna_allgenes_results_validation.csv",
                     index=False)
all_val_df.to_csv(f"../results/tcga_brca_subtype/lr_all_res_cnv_rna_allgenes_results_validation.csv.gz", index=False)

# print("Running RF")
# rf_results_df, val_res_perclass = run_model(input_df_train, input_df_test, "RF")
# all_val_df.columns = [id_to_class_name[int(x.split("_")[-1])] if "feature_" in x else x for x in
#                           all_val_df.columns]
# rf_results_df.to_csv("../results/tcga_brca_subtype/rf_cnv_rna_allgenes_results_validation.csv",
#                      index=False)
# all_val_df.to_csv(f"../results/tcga_brca_subtype/rf_all_res_cnv_rna_allgenes_results_validation.csv.gz", index=False)

# print("Running XGB")
# xgb_results_df, val_res_perclass = run_model(input_df_train, input_df_test, "XGB")
# xgb_results_df.to_csv("../results/tcga_brca_subtype/xgb_cnv_rna_allgenes_validation.csv",
#                       index=False)

