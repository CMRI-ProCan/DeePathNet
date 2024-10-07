library(mixOmics)
library(Metrics)

# input_data = "sanger"
# target = "gdsc"

# args = commandArgs(trailingOnly=TRUE)
# 
# input_data = args[1]
# target = args[2]
# mode = args[3]

input_data = "ccle"
target = "ctd2"
mode = "allgenes"

if (input_data == "sanger"){
  mutation = read.csv("./data/processed/omics/sanger_df_mutation_drug.csv.gz", row.names = 1)
  cnv = read.csv("./data/processed/omics/sanger_df_cnv_drug.csv.gz", row.names = 1)
  rna = read.csv("./data/processed/omics/sanger_df_rna_drug.csv.gz", row.names = 1)
  train_splits = read.csv("./data/meta/25splits_train_sanger.csv", stringsAsFactors = F)
  val_splits = read.csv("./data/meta/25splits_val_sanger.csv", stringsAsFactors = F)
  
} else {
  if (target == "gdsc"){
    train_splits = read.csv("./data/meta/25splits_train_ccle_gdsc.csv", stringsAsFactors = F)
    val_splits = read.csv("./data/meta/25splits_val_ccle_gdsc.csv", stringsAsFactors = F)
    mutation = read.csv("./data/processed/omics/ccle_df_mutation_drug.csv.gz", row.names = 1)
    cnv = read.csv("./data/processed/omics/ccle_df_cnv_drug.csv.gz", row.names = 1)
    rna = read.csv("./data/processed/omics/ccle_df_rna_drug.csv.gz", row.names = 1)
  } else {
    mutation = read.csv("./data/processed/omics/ccle_df_mutation_ctd2.csv.gz", row.names = 1)
    cnv = read.csv("./data/processed/omics/ccle_df_cnv_ctd2.csv.gz", row.names = 1)
    rna = read.csv("./data/processed/omics/ccle_df_rna_ctd2.csv.gz", row.names = 1)
    train_splits = read.csv("./data/meta/25splits_train_ccle_ctd2.csv", stringsAsFactors = F)
    val_splits = read.csv("./data/meta/25splits_val_ccle_ctd2.csv", stringsAsFactors = F)
  }
}

if (target == "gdsc"){
  if (input_data == "sanger"){
    target_data = read.csv("./data/processed/drug/sanger_gdsc_intersection_noprot_wide.csv.gz", row.names = 1)
  } else {
    target_data = read.csv("./data/processed/drug/ccle_gdsc_intersection3_wide.csv.gz", row.names = 1)
  }
} else {
  if (input_data == "sanger"){
    target_data = read.csv("./data/processed/drug/sanger_ctd2_min400.csv.gz", row.names = 1)
  } else {
    target_data = read.csv("./data/processed/drug/ccle_ctd2_min600.csv.gz", row.names = 1)
  }
  
}

print(input_data)
print(target)

drug_samples = rownames(target_data)
target_data = sapply(target_data, as.numeric)
rownames(target_data) = drug_samples

lc_cancer_genes = read.csv("./data/meta/lc_cancer_genes.csv", stringsAsFactors = F)



common_samples = intersect(rownames(mutation), drug_samples)

if (mode != "allgenes"){
  mutation = mutation[common_samples, colnames(mutation) %in% lc_cancer_genes$cancer_genes]
  cnv = cnv[common_samples, colnames(cnv) %in% lc_cancer_genes$cancer_genes]
  rna = rna[common_samples, colnames(rna) %in% lc_cancer_genes$cancer_genes]
}

target_data = target_data[common_samples,]


res_df <- data.frame(drug_id=character(),
                     run=character(),
                     corr=numeric(),
                     r2=numeric(),
                     mae=numeric(),
                     rmse=numeric(),
                     time=numeric(),
                     stringsAsFactors=FALSE)

for (i in 1:25){
  train_samples = intersect(train_splits[, i], drug_samples)
  val_samples = intersect(val_splits[, i], drug_samples)
  
  mutation_train = mutation[rownames(mutation) %in% train_samples,]
  mutation_val = mutation[rownames(mutation) %in% val_samples,]
  
  cnv_train = cnv[rownames(cnv) %in% train_samples,]
  cnv_val = cnv[rownames(cnv) %in% val_samples,]
  
  rna_train = rna[rownames(rna) %in% train_samples,]
  rna_val = rna[rownames(rna) %in% val_samples,]
  
  train_data = list(mutation = mutation_train,
                    CNV = cnv_train,
                    RNA = rna_train)
  
  val_data = list(mutation = mutation_val,
                  CNV = cnv_val,
                  RNA = rna_val)
  
  start_time <- Sys.time()
  N_comp = 50
  model = block.spls(train_data, target_data[rownames(target_data) %in% train_samples, ], ncomp = N_comp)
  target_val = target_data[rownames(target_data) %in% val_samples, ]
  pred = predict(model, val_data)
  pred_values = pred$AveragedPredict[,,N_comp]
  end_time <- Sys.time()
  time_taken = as.numeric(end_time - start_time)
  
  for (drug_idx in 1:dim(target_data)[2]){
    curr_pred = pred_values[,drug_idx]
    curr_target = target_val[,drug_idx]
    pred_mean = mean(na.omit(curr_pred))
    curr_pred[is.na(curr_pred)] = pred_mean
    curr_pred = curr_pred[!is.na(curr_target)]
    curr_target = curr_target[!is.na(curr_target)]
    pcorr = cor(curr_pred, curr_target, method="pearson")
    r2 =  1 - sum((curr_target - curr_pred)^2) / sum((curr_target - mean(curr_target))^2)
    mae = mean(abs(curr_target - curr_pred))
    rmse_drug = rmse(curr_target, curr_pred)
    
    next_row = c(colnames(target_data)[drug_idx], paste0("cv_", i-1), pcorr, r2, mae,rmse_drug, time_taken)
    res_df[nrow(res_df) + 1,] = next_row
  }
  
  print(paste0("run ", i, " completed"))
  flush.console()
}

filename = paste0("./work_dirs/mixOmics/", input_data, "_mutation_cnv_rna_", target, "_", mode, "_", N_comp, "_comp.csv")
write.csv(res_df, filename, row.names = F)
