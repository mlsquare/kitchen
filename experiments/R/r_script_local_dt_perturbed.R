library(tidyr)
library(dplyr)
library(hash)
library(rpart)
library("rpart.plot")
library(purrr)

set.seed(42) 

## Set working directory
setwd('/Users/nayana/projects/Project/kitchen/experiments/')

## Load functions
source("R/dataset_functions.R")

## Set dataset
dataset_name <- "iris"

## Load dataset
dataset_csv <- dataset_source_csv(dataset_name)
dataset <- read.csv(dataset_csv, header = TRUE, row.names = NULL)
col_names <- dataset_col_names(dataset_name)
colnames(dataset) <- col_names
dataset_p_csv <- dataset_perturbed_csv(dataset_name)
dataset_p <- read.csv(dataset_p_csv, header = TRUE, row.names = NULL)
dataset_p <- dataset_p[,1:(length(dataset_p)-1)]
colnames(dataset_p) <- col_names

## Preprocessing
dataset <- dataset_preprocessed(dataset_name, dataset)

## Target
target_var <- col_names[length(col_names)]

## Add row IDs
ID <- 1:nrow(dataset)
dataset <- cbind(ID, dataset)

ID <- 1:nrow(dataset_p)
dataset_p <- cbind(ID, dataset_p)

## Traversing tree using an instance from input data(x)
return_path <- function(x){
  node <- 1
  nspl <- 1
  path_list <- list()
  test_yval <- list()
  while(nspl != 0){
    i <- subset(x, select=-c(species))
    npos <- match(node, nnum)
    nspl <- nodes[4][npos,] # Recheck
    var <- vnum[nspl]
    ncat <- split_df[2][nspl,]
    temp <- split_df[4][nspl,]
    if (nspl > 0){
      if (ncat >= 2){
        dir = csplit[temp,as.integer(i[var])]
        label <- ifelse(dir == -1, 0, 1)
        level <- paste("(", as.character(i[var]), ")", sep = "")
        if (!(level %in% keys(invert(global_bins)))){
          .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=level)
        }
        
        node_decision <- paste(as.character(var), invert(global_bins)[[level]], label, sep = "")
        path_list <- c(path_list, node_decision)
      }
      else if (i[var] < temp){
        if (!(as.character(temp) %in% keys(invert(global_bins)))){
          .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=as.character(temp))
        }
        label <- invert(global_bins)[[as.character(temp)]]
        node_decision <- paste(as.character(var), label, 0, sep = "")
        path_list <- c(path_list, node_decision)
        dir = ncat
      }
      else {
        if (!(as.character(temp) %in% keys(invert(global_bins)))){
          .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=as.character(temp))
        }
        label <- invert(global_bins)[[as.character(temp)]]
        node_decision <- paste(as.character(var), label, 1, sep = "")
        path_list <- c(path_list, node_decision)
        dir = -ncat
      }
      
      if (dir == -1){
        node = 2 * node
      }
      else{
        node = 2 * node + 1
      }
    }
    else{
    }
  }
  path_list = paste(path_list, collapse = ",")
  names(path_list) <- "new_col"
  return(path_list)
}

# Number of trees constructed (1 - 1000)
ntrees <- get_ntrees(dataset_name)     
# First row for which tree is constructed
start <- get_start(dataset_name) 
# Number of rows considered
nsample <- get_nsample(dataset_name)  
# Number of times primary instance is included in the local tree
nprim <- get_nprim(dataset_name)
# Number of secondary instances included in the local tree
nsec <- get_nsec(dataset_name)

dataset <- dataset[start:(start-1+ntrees),]

## Local dt and paths list setup
final_paths <- data.frame(matrix(ncol = 2, nrow = nsample)) #original and perturbed

set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
names(set1) <- "new_col"
set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
bin_labels <- bind_rows(set1, set2, set3)

global_bins <- hash()

for (iter_index in 1:ntrees) {
  print(iter_index)
  x <- dataset[iter_index, 1:ncol(dataset)]
  x <- x[rep(seq_len(nrow(x)), nprim),]
  x <- x[rowSums(is.na(x)) == 0,]  
  x <- rbind(x, dataset %>%
               slice(-c(iter_index))%>%
               slice(sample(nrow(dataset)))%>%
               slice(1:nsec))
  instances <- x[,c("ID")]
  x <- x[,2:ncol(dataset)]
  
  ## Create local tree and export details as csv
  f <- paste0(target_var," ~ .")
  local_tree <- rpart(f, data = x, minsplit=1, cp=0)
  model <- paste0("R/trees/",dataset_name,"_perturbed_local_tree",iter_index,".RData")
  save(local_tree, file = model)
  
  frame_csv_name <- paste0("R/local_dt_info_", dataset_name, "_perturbed/frame", "_", iter_index,".csv")
  splits_csv_name <- paste0("R/local_dt_info_", dataset_name, "_perturbed/splits", "_", iter_index,".csv")
  #write.csv(local_tree$frame, file=frame_csv_name)
  #write.csv(local_tree$splits, file=splits_csv_name)
  temp_frame <- local_tree$frame
  
  nodes <- as.numeric(rownames(temp_frame))

  nc <- temp_frame[, c("ncompete", "nsurrogate")]
  
  temp_frame$index <- 1L + c(0L, cumsum((temp_frame$var != "<leaf>") + nc[[1L]] + nc[[2L]]))[-(nrow(temp_frame) + 1L)] ## Validate the values!
  
  temp_frame$index[temp_frame$var == "<leaf>"] <- 0L
  
  nodes <- temp_frame[, c("n", "ncompete", "nsurrogate", "index")]
  
  nnum = row.names(temp_frame)
  
  vnum <- match(rownames(local_tree$splits), colnames(dataset[1,2:ncol(dataset)]))
  
  split <- local_tree$splits
  
  split_rownames <- rownames(split)
  
  split_df <- as.data.frame(split, row.names = 0)
  
  csplit <- local_tree$csplit -2L
  
  path_list <- return_path(dataset[iter_index,2:ncol(dataset)])
  final_paths[iter_index, 1] <- path_list
}

for (iter_index in 1:ntrees) {
  print(iter_index)
  
  model <- paste0("R/trees/",dataset_name,"_perturbed_local_tree",iter_index,".RData")
  local_tree <- get(load(file = model))
  
  temp_frame <- local_tree$frame
  
  nc <- temp_frame[, c("ncompete", "nsurrogate")]
  
  temp_frame$index <- 1L + c(0L, cumsum((temp_frame$var != "<leaf>") + nc[[1L]] + nc[[2L]]))[-(nrow(temp_frame) + 1L)] ## Validate the values!
  
  temp_frame$index[temp_frame$var == "<leaf>"] <- 0L
  
  nodes <- temp_frame[, c("n", "ncompete", "nsurrogate", "index")]
  
  nnum = row.names(temp_frame)
  
  vnum <- match(rownames(local_tree$splits), colnames(dataset[1,2:ncol(dataset)]))
  
  split <- local_tree$splits
  
  split_rownames <- rownames(split)
  
  split_df <- as.data.frame(split, row.names = 0)
  
  csplit <- local_tree$csplit -2L
  
  path_list_p <- return_path(dataset_p[iter_index,2:ncol(dataset)])
  final_paths[iter_index, 2] <- path_list_p
}

# Primary paths - original and perturbed
col_headings <- c('original','perturbed')
names(final_paths) <- col_headings
#write.csv(final_paths, file = paste0("Outputs/", dataset_name, "_perturbed_paths_", ntrees, ".csv"), row.names = FALSE)

label_df <- data.frame(values(global_bins))
#write.csv(label_df, file = paste0("Outputs/", dataset_name, "_perturbed_bin_labels_", ntrees, ".csv"))