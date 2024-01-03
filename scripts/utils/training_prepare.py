import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_logger(config_file, STAMP):
    configs = json.load(open(config_file, "r"))
    log_suffix = ""
    if "suffix" in configs:
        log_suffix = configs["suffix"]
    log_file = f"{STAMP}{log_suffix}.log"
    logger = logging.getLogger("multi-drug")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(configs["work_dir"], log_file))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(open(config_file, "r").read())
    print(open(config_file, "r").read())
    return logger


def prepare_data_cv(config_file, STAMP):
    configs = json.load(open(config_file, "r"))

    if not os.path.isdir(configs["work_dir"]):
        os.system(f"mkdir -p {configs['work_dir']}")

    data_file = configs["data_file"]
    data_type = configs["data_type"]

    data_target = pd.read_csv(configs["target_file"], low_memory=False, index_col=0)
    if "drug_id" in configs and configs["drug_id"] != "":
        data_target = data_target[configs["drug_id"]]

    data_input = pd.read_csv(data_file, index_col=0)
    if data_type[0] != "DR":
        data_input = data_input[
            [
                x
                for x in data_input.columns
                if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)
            ]
        ]

    if "downsample" in configs:
        _, test_idx = train_test_split(
            data_input.index,
            test_size=configs["downsample"],
            random_state=42,
            stratify=data_target,
        )
        data_input = data_input.loc[test_idx]
        data_target = data_target.loc[test_idx]
        # data_input = data_input.sample(n=configs["downsample"], random_state=42)
        # data_target = data_target.loc[data_input.index]

    tissues = (
        [x for x in data_input.columns if "tissue" in x]
        if "tissue" in data_type
        else None
    )
    omics_types = [x for x in data_type if x != "tissue"]
    with_tissue = "tissue" in data_type

    genes = np.unique(
        ([x.split("_")[0] for x in data_input.columns if x.split("_")[0] != "tissue"])
    )

    num_of_features = data_input.shape[1]

    if configs["task"] == "regression":
        val_score_dict = {
            "drug_id": [],
            "run": [],
            "epoch": [],
            "mae": [],
            "rmse": [],
            "corr": [],
            "r2": [],
        }
    elif configs["task"] == "multiclass":
        val_score_dict = {
            "run": [],
            "epoch": [],
            "top1_acc": [],
            "top3_acc": [],
            "f1": [],
            "roc_auc": [],
        }
    else:
        val_score_dict = {
            "drug_id": [],
            "run": [],
            "epoch": [],
            "accuracy": [],
            "auc": [],
        }

    ret_dict = {
        "data_input_all": data_input,
        "data_target_all": data_target,
        "val_score_dict": val_score_dict,
        "num_of_features": num_of_features,
        "genes": genes,
        "omics_types": omics_types,
        "with_tissue": with_tissue,
        "tissues": tissues,
        "logger": logger,
    }
    return ret_dict


def prepare_data_independent_test(config_file, STAMP, seed=1):
    configs = json.load(open(config_file, "r"))

    if not os.path.isdir(configs["work_dir"]):
        os.system(f"mkdir -p {configs['work_dir']}")

    data_file_train = configs["data_file_train"]
    data_file_test = configs["data_file_test"]
    data_type = configs["data_type"]

    data_target_train = pd.read_csv(
        configs["target_file_train"], low_memory=False, index_col=0
    )
    data_target_test = pd.read_csv(
        configs["target_file_test"], low_memory=False, index_col=0
    )

    if "drug_id" in configs and configs["drug_id"] != "":
        data_target_train = data_target_train[configs["drug_id"]]

    data_input_train = pd.read_csv(data_file_train, index_col=0).fillna(0)
    data_input_test = pd.read_csv(data_file_test, index_col=0).fillna(0)
    common_features = list(
        set(data_input_train.columns).intersection(data_input_test.columns)
    )
    common_cells_train = list(
        set(data_input_train.index).intersection(data_target_train.index)
    )
    common_cells_test = list(
        set(data_input_test.index).intersection(data_target_test.index)
    )
    data_input_train = data_input_train.loc[common_cells_train, common_features]
    data_target_train = data_target_train.loc[common_cells_train]
    data_input_test = data_input_test.loc[common_cells_test, common_features]
    data_target_test = data_target_test.loc[common_cells_test]

    if data_type[0] != "DR":
        data_input_train = data_input_train[
            [
                x
                for x in data_input_train.columns
                if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)
            ]
        ]
        data_input_test = data_input_test[
            [
                x
                for x in data_input_test.columns
                if (x.split("_")[1] in data_type) or (x.split("_")[0] in data_type)
            ]
        ]

    if "downsample" in configs:
        _, test_idx = train_test_split(
            data_input_train.index,
            test_size=configs["downsample"],
            random_state=seed,
        )
        data_input_train = data_input_train.loc[test_idx]
        data_target_train = data_target_train.loc[test_idx]

    tissues = (
        [x for x in data_input_train.columns if "tissue" in x]
        if "tissue" in data_type
        else None
    )
    omics_types = [x for x in data_type if x != "tissue"]
    with_tissue = "tissue" in data_type

    genes = np.unique(
        (
            [
                x.split("_")[0]
                for x in data_input_train.columns
                if x.split("_")[0] != "tissue"
            ]
        )
    )

    num_of_features = data_input_train.shape[1]

    ret_dict = {
        "data_input_train": data_input_train,
        "data_target_train": data_target_train,
        "data_input_test": data_input_test,
        "data_target_test": data_target_test,
        "num_of_features": num_of_features,
        "genes": genes,
        "omics_types": omics_types,
        "with_tissue": with_tissue,
        "tissues": tissues,
    }
    return ret_dict


def get_score_dict(config_file):
    configs = json.load(open(config_file, "r"))
    if configs["task"] == "regression":
        val_score_dict = {
            "drug_id": [],
            "run": [],
            "epoch": [],
            "mae": [],
            "rmse": [],
            "corr": [],
            "r2": [],
        }
    elif configs["task"] == "multiclass":
        val_score_dict = {
            "run": [],
            "epoch": [],
            "top1_acc": [],
            "top3_acc": [],
            "f1": [],
            "roc_auc": [],
        }
    else:
        val_score_dict = {
            "drug_id": [],
            "run": [],
            "epoch": [],
            "accuracy": [],
            "auc": [],
        }
    return val_score_dict
