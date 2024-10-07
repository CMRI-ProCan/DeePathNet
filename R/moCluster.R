library(mogsa)

# mutation = read.csv("./data/processed/omics/ccle_df_mutation_drug.csv.gz", row.names = 1)
# cnv = read.csv("./data/processed/omics/ccle_df_cnv_drug.csv.gz", row.names = 1)
# rna = read.csv("./data/processed/omics/ccle_df_rna_drug.csv.gz", row.names = 1)

# mutation = read.csv("./data/processed/omics/ccle_df_mutation_drug_ctd2.csv.gz", row.names = 1)
# cnv = read.csv("./data/processed/omics/ccle_df_cnv_drug_ctd2.csv.gz", row.names = 1)
# rna = read.csv("./data/processed/omics/ccle_df_rna_drug_ctd2.csv.gz", row.names = 1)

mutation = read.csv("./data/processed/omics/tcga_23_cancer_types_mutation.csv.gz", row.names = 1)
cnv = read.csv("./data/processed/omics/tcga_23_cancer_types_cnv.csv.gz", row.names = 1)
rna = read.csv("./data/processed/omics/tcga_23_cancer_types_rna.csv.gz", row.names = 1)
mutation[is.na(mutation)] <- 0
cnv[is.na(cnv)] <- 0
rna[is.na(rna)] <- 0

lc_cancer_genes = read.csv("./data/meta/lc_cancer_genes.csv", stringsAsFactors = F)
lc_cancer_genes$cancer_genes

common_samples = rownames(mutation)

# mutation = mutation[, colnames(mutation) %in% lc_cancer_genes$cancer_genes]
# cnv = cnv[, colnames(cnv) %in% lc_cancer_genes$cancer_genes]
# rna = rna[, colnames(rna) %in% lc_cancer_genes$cancer_genes]

mutation <- sapply(mutation, as.numeric)
cnv <- sapply(cnv, as.numeric)
rna <- sapply(rna, as.numeric)

mo.combined = list(t(mutation), t(cnv), t(rna))
# mo.combined = list(t(cnv), t(rna), t(protein))
# mo.combined = list(t(methy), t(rna), t(protein))
# mo.combined = list(t(rna), t(protein))
print(length(mo.combined))

moa <- mbpca(mo.combined, ncomp = 200, k = "all", method = "globalScore", 
             option = "lambda1", center=TRUE, scale=FALSE, moa = TRUE, 
             svd.solver = "fast", maxiter = 1000)

res.df = as.data.frame(moa@fac.scr)
res.df$Cell_line = common_samples
res.df = res.df[, c(dim(res.df)[2], 1:(dim(res.df)[2]-1))]
write.table(res.df, "./data/DR/moCluster/tcga_23_cancer_types_mutation_cnv_rna_allgenes.csv", sep = ",", quote = F, row.names = F)

# tmp = read.csv("./data/DR/moCluster/ccle_mutation_cnv_rna_gdsc.csv", sep = ",")
# tmp = tmp[, c(2:(dim(tmp)[2]-1), dim(tmp)[2])]
# res.df = tmp
