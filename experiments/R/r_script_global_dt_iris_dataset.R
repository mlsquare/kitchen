library(tidyr)
library(dplyr)
library(hash)
library(rpart)
## Load dataset
iris_dataset <- read.csv("../Data/iris_original.csv", header = FALSE)
col_names <- c('sepal_length','sepal_width','petal_length','petal_width','class')
colnames(iris_dataset) <- col_names

set.seed(42)

## Traversing tree using an instance from input data(x)
return_path <- function(x){
  node <- 1
  nspl <- 1
  path_list <- list()
  test_yval <- list()
  while(nspl != 0){
    i <- subset(x, select=-c(class))
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

## global dt and paths list setup
final_paths <- data.frame(matrix(ncol = 1, nrow = 150)) 

set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
names(set1) <- "new_col"
set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
bin_labels <- bind_rows(set1, set2, set3)

global_bins <- hash()

x <- iris_dataset

## Create global tree and export details as csv
global_tree <- rpart(class ~ ., data = x, minsplit=1,cp=0)
frame_csv_name <- paste0("global_dt_info_iris/frame", "_", iter_index,".csv")
splits_csv_name <- paste0("global_dt_info_iris/splits", "_", iter_index,".csv")
write.csv(global_tree$frame, file=frame_csv_name)
write.csv(global_tree$splits, file=splits_csv_name)

temp_frame <- global_tree$frame

nc <- temp_frame[, c("ncompete", "nsurrogate")]

temp_frame$index <- 1L + c(0L, cumsum((temp_frame$var != "<leaf>") + nc[[1L]] + nc[[2L]]))[-(nrow(temp_frame) + 1L)] ## Validate the values!

temp_frame$index[temp_frame$var == "<leaf>"] <- 0L

nodes <- temp_frame[, c("n", "ncompete", "nsurrogate", "index")]

nnum = row.names(temp_frame)

vnum <- match(rownames(global_tree$splits), colnames(iris_dataset[1,]))

split <- global_tree$splits

split_rownames <- rownames(split)

split_df <- as.data.frame(split, row.names = 0)

csplit <- global_tree$csplit -2L

for (iter_index in 1:150) {
  print(iter_index)
  path_list <- return_path(iris_dataset[iter_index,1:5])
  final_paths[iter_index, ] <- path_list
  #write.table(path<-list, file = "<NAME>.csv", sep = ",", append = TRUE, quote = FALSE, col.names = paste('X',toString(iter_index)), row.names = FALSE)
}

## Export paths and bin details
write.csv(final_paths, file="../Outputs/test_iris_global_paths_100.csv")
label_df <- data.frame(values(global_bins))
write.csv(label_df, file="../Outputs/test_iris_global_bin_labels_100.csv")