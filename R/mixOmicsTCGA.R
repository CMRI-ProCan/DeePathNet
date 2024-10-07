library(mixOmics)
library(pROC)
library(mltest)

# input_data = "tcga_23_cancer_types"
# input_data = "tcga_brca"

# target = "all"
# target = "brca_subtypes"

args = commandArgs(trailingOnly=TRUE)

input_data = args[1]

mode = args[2]

if (input_data == "tcga_23_cancer_types") {
  mutation = read.csv("./data/processed/omics/tcga_23_cancer_types_mutation.csv.gz", row.names = 1)
  cnv = read.csv("./data/processed/omics/tcga_23_cancer_types_cnv.csv.gz", row.names = 1)
  rna = read.csv("./data/processed/omics/tcga_23_cancer_types_rna.csv.gz", row.names = 1)
  target_data = read.csv("./data/processed/cancer_type/tcga_23_cancer_types_mutation_cnv_rna.csv", row.names = 1)
  train_splits = read.csv("./data/meta/25splits_train_tcga.csv", stringsAsFactors = F)
  val_splits = read.csv("./data/meta/25splits_val_tcga.csv", stringsAsFactors = F)
} else {
  mutation = read.csv("./data/processed/omics/tcga_brca_mutation.csv.gz", row.names = 1)
  cnv = read.csv("./data/processed/omics/tcga_brca_cnv.csv.gz", row.names = 1)
  rna = read.csv("./data/processed/omics/tcga_brca_rna.csv.gz", row.names = 1)
  target_data = read.csv("./data/processed/cancer_type/tcga_brca_mutation_cnv_rna_subtypes.csv", row.names = 1)
  train_splits = read.csv("./data/meta/25splits_train_tcga_brca.csv", stringsAsFactors = F)
  val_splits = read.csv("./data/meta/25splits_val_tcga_brca.csv", stringsAsFactors = F)
}

mutation[is.na(mutation)] <- 0
cnv[is.na(cnv)] <- 0
rna[is.na(rna)] <- 0
samples = rownames(target_data)
target_data = sapply(target_data, as.numeric)
rownames(target_data) = samples

lc_cancer_genes = read.csv("./data/meta/lc_cancer_genes.csv", stringsAsFactors = F)



common_samples = intersect(rownames(mutation), samples)

# mutation = mutation[common_samples, colnames(mutation) %in% lc_cancer_genes$cancer_genes]
# cnv = cnv[common_samples, colnames(cnv) %in% lc_cancer_genes$cancer_genes]
# rna = rna[common_samples, colnames(rna) %in% lc_cancer_genes$cancer_genes]
target_data = target_data[common_samples,]


res_df <- data.frame(run=character(), 
                     top1_acc=numeric(),
                     f1=numeric(),
                     roc_auc=numeric(),
                     time=numeric(),
                     stringsAsFactors=FALSE)
i=1
for (i in 1:25){
  train_samples = intersect(train_splits[, i], samples)
  val_samples = intersect(val_splits[, i], samples)
  
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
  model = block.splsda(train_data, target_data[names(target_data) %in% train_samples], ncomp = N_comp)
  target_val = target_data[names(target_data) %in% val_samples]
  pred = predict(model, val_data)
  pred_values = as.numeric(pred$AveragedPredict.class$max.dist[,N_comp])
  end_time <- Sys.time()
  time_taken = as.numeric(end_time - start_time)
  
  auc = multiclass.roc(target_val,pred_values)$auc
  
  y_true = as.factor(target_val)
  y_pred = as.factor(pred_values)
  levels(y_pred) = levels(y_true)
  test_res = ml_test(y_pred, y_true, output.as.table = FALSE)
  f1_macro = mean(na.omit(test_res$F1))
  accuracy = test_res$accuracy
  
  next_row = c(paste0("cv_", i-1), accuracy, f1_macro, auc, time_taken)
  res_df[nrow(res_df) + 1,] = next_row
  
  print(paste0("run ", i, " completed"))
  flush.console()
}

filename = paste0("./work_dirs/mixOmics/", input_data, "_mutation_cnv_rna_" , mode, "_", N_comp, "_comp.csv")
write.csv(res_df, filename, row.names = F)
