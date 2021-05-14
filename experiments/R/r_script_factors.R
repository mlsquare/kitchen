set.seed(42)

## Set working directory
setwd('/Users/nayana/projects/Project/kitchen/experiments/')

## Load functions
source("R/dataset_functions.R")

## Set dataset
dataset_name <- "adult"

## Load dataset
dataset_csv <- dataset_source_csv(dataset_name)
dataset <- read.csv(dataset_csv, header = TRUE, row.names = NULL)
col_names <- dataset_col_names(dataset_name)
colnames(dataset) <- col_names

factor_cols <- names(Filter(is.factor, dataset))

l <- length(factor_cols)

counts <- dataset %>% 
  lapply(levels) %>% 
  lapply(table) %>%
  lapply(length)

maxval <- max(unlist(counts)) + 1

factors <- matrix(ncol = maxval, nrow = l)

for( i in 1:l){
  new_row <- c(match(factor_cols[i],names(dataset)), levels(dataset[,factor_cols[i]]))
  length(new_row) <- maxval
  factors[i,] <- new_row
}

factors_df <- data.frame(t(factors))
names(factors_df) <- factor_cols
write.csv(factors_df, file=paste0("Outputs/",dataset_name,"_dataset_factors.csv"), row.names = FALSE, na = "")