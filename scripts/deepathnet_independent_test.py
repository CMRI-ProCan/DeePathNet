"""
Script to run DeePathNet with independent test set for any task.
E.g. python scripts/deepathnet_independent_test.py configs/sanger_train_ccle_test_gdsc/mutation_cnv_rna_prot/deepathnet_mutation_cnv_rna_prot.json
"""
import json
import sys
from datetime import datetime
import numpy as np
import torch.optim
from torch.utils.data import DataLoader

from utils.training_prepare import (
    prepare_data_independent_test,
    get_logger,
    get_score_dict,
)

from models import *
from model_transformer_lrp import DeePathNet

STAMP = datetime.today().strftime("%Y%m%d%H%M")

config_file = sys.argv[1]
# load model configs
configs = json.load(open(config_file, "r"))

log_suffix = ""
if "suffix" in configs:
    log_suffix = configs["suffix"]

seed = configs["seed"]
torch.manual_seed(seed)
np.random.seed(seed)

BATCH_SIZE = configs["batch_size"]
NUM_WORKERS = 0
LOG_FREQ = configs["log_freq"]
NUM_EPOCHS = configs["num_of_epochs"]
device = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_CONTROL = False if "random_control" not in configs else configs["random_control"]

logger = get_logger(config_file, STAMP)


def get_setup(genes_to_id, id_to_genes, target_dim, cv=0):
    def load_pathway(random_control=False):
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
            logger.info(
                f"Filtering pathway with GeneNumber < {configs['max_gene_num']}"
            )
        if "min_gene_num" in configs:
            pathway_df = pathway_df[pathway_df["GeneNumber"] > configs["min_gene_num"]]
            logger.info(
                f"Filtering pathway with GeneNumber > {configs['min_gene_num']}"
            )

        pathway_df["genes"] = pathway_df["genes"].map(
            lambda x: "|".join([gene for gene in x.split("|") if gene in genes])
        )

        for index, row in pathway_df.iterrows():
            if row["genes"]:
                pathway_dict[row["name"]] = row["genes"].split("|")
        cancer_genes = set(
            [y for x in pathway_df["genes"].values for y in x.split("|")]
        )
        non_cancer_genes = set(genes) - set(cancer_genes)
        logger.info(
            f"Cancer genes:{len(cancer_genes)}\tNon-cancer genes:{len(non_cancer_genes)}"
        )
        if random_control:
            logger.info("Randomly select genes for each pathway")
            for key in pathway_dict:
                pathway_dict[key] = list(
                    np.random.choice(
                        list(set(cancer_genes)), len(pathway_dict[key]), replace=False
                    )
                )
        return pathway_dict, non_cancer_genes

    pathway_dict, non_cancer_genes = load_pathway(random_control=RANDOM_CONTROL)
    if RANDOM_CONTROL:
        logger.info("Saving random control genes")
        with open(
            f"{configs['work_dir']}/random_genes_cv{cv}_{STAMP}{log_suffix}.json", "w"
        ) as f:
            json.dump(pathway_dict, f)

    model = DeePathNet(
        len(omics_types),
        target_dim,
        genes_to_id,
        id_to_genes,
        pathway_dict,
        non_cancer_genes,
        embed_dim=configs["dim"],
        depth=configs["depth"],
        mlp_ratio=configs["mlp_ratio"],
        out_mlp_ratio=configs["out_mlp_ratio"],
        num_heads=configs["heads"],
        pathway_drop_rate=configs["pathway_dropout"],
        only_cancer_genes=configs["cancer_only"],
        tissues=tissues,
    )
    logger.info(
        open("/home/scai/DeePathNet/scripts/model_transformer_lrp.py", "r").read()
    )

    logger.info(model)
    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )

    logger.info(optimizer)

    lr_scheduler = None

    return model, criterion, optimizer, lr_scheduler


def run_experiment(
    merged_df_train,
    merged_df_test,
    val_score_dict,
    run="test",
    class_name_to_id=None,
    cv=0,
):
    train_df = merged_df_train.iloc[:, :num_of_features]
    test_df = merged_df_test.iloc[:, :num_of_features]
    train_target = merged_df_train.iloc[:, num_of_features:]
    test_target = merged_df_test.iloc[:, num_of_features:]

    X_train = train_df
    X_test = test_df

    if configs["task"] == "multiclass":
        train_dataset = MultiOmicMulticlassDataset(
            X_train,
            train_target,
            mode="train",
            omics_types=omics_types,
            class_name_to_id=class_name_to_id,
            logger=logger,
        )
        test_dataset = MultiOmicMulticlassDataset(
            X_test,
            test_target,
            mode="val",
            omics_types=omics_types,
            class_name_to_id=class_name_to_id,
            logger=logger,
        )
    else:
        train_dataset = MultiOmicDataset(
            X_train,
            train_target,
            mode="train",
            omics_types=omics_types,
            logger=logger,
            with_tissue=with_tissue,
        )
        test_dataset = MultiOmicDataset(
            X_test,
            test_target,
            mode="val",
            omics_types=omics_types,
            logger=logger,
            with_tissue=with_tissue,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=configs["drop_last"],
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    if configs["task"] == "multiclass":
        target_dim = len(class_name_to_id)
    else:
        target_dim = train_target.shape[1]
    model, criterion, optimizer, lr_scheduler = get_setup(
        train_dataset.genes_to_id, train_dataset.id_to_genes, target_dim, cv=cv
    )

    val_drug_ids = merged_df_test.columns[num_of_features:]
    val_res = train_loop(
        NUM_EPOCHS,
        train_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        logger,
        STAMP,
        configs,
        lr_scheduler,
        val_drug_ids,
        run=run,
        val_score_dict=val_score_dict,
    )

    return val_res


num_repeat = 1 if "num_repeat" not in configs else configs["num_repeat"]
count = 0
all_val_df = []
val_score_dict = get_score_dict(config_file)
for n in range(num_repeat):
    data_dict = prepare_data_independent_test(config_file, STAMP, seed=count)
    data_input_train = data_dict["data_input_train"]
    data_target_train = data_dict["data_target_train"]
    data_input_test = data_dict["data_input_test"]
    data_target_test = data_dict["data_target_test"]
    num_of_features = data_dict["num_of_features"]
    genes = data_dict["genes"]
    omics_types = data_dict["omics_types"]
    with_tissue = data_dict["with_tissue"]
    tissues = data_dict["tissues"]

    class_name_to_id = None
    id_to_class_name = None
    if configs["task"] == "multiclass":
        class_name_to_id = dict(
            zip(
                sorted(data_target_train.iloc[:, 0].unique()),
                list(range(data_target_train.iloc[:, 0].unique().size)),
            )
        )
        id_to_class_name = dict(
            zip(
                list(range(data_target_train.iloc[:, 0].unique().size)),
                sorted(data_target_train.iloc[:, 0].unique()),
            )
        )

    merged_df_train = pd.merge(data_input_train, data_target_train, on=["Cell_line"])
    merged_df_test = pd.merge(data_input_test, data_target_test, on=["Cell_line"])

    val_res = run_experiment(
        merged_df_train,
        merged_df_test,
        val_score_dict,
        run=f"cv_{count}",
        class_name_to_id=class_name_to_id,
        cv=count,
    )
    all_val_df.append(val_res)
    count += 1

if "save_scores" not in configs or configs["save_scores"]:
    val_score_df = pd.DataFrame(val_score_dict)
    val_score_df.to_csv(
        f"{configs['work_dir']}/scores_{STAMP}{log_suffix}.csv.gz", index=False
    )
if configs["task"] == "multiclass":
    all_val_df = pd.concat(all_val_df)
    all_val_df["y_pred"] = all_val_df["y_pred"].map(id_to_class_name)
    all_val_df["y_true"] = all_val_df["y_true"].map(id_to_class_name)
    all_val_df.columns = [
        id_to_class_name[int(x.split("_")[-1])] if "feature_" in x else x
        for x in all_val_df.columns
    ]
    all_val_df.to_csv(
        f"{configs['work_dir']}/all_val_res_{STAMP}{log_suffix}.csv.gz", index=False
    )

logger.info("Full training finished.")
