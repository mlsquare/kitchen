
library(tidyr)
library(dplyr)
library(hash)
library(rpart)
## Load dataset
adult_dataset <- read.csv("../../data/raw/adult.data.csv", header = FALSE)
col_names <- c('age', 'workclass', 'fnlwgt', 'education', 'education-num',
         'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'target')
colnames(adult_dataset) <- col_names

adult_dataset = adult_dataset[adult_dataset["workclass"] != " ?",]
adult_dataset = adult_dataset[adult_dataset["occupation"] != " ?",]
adult_dataset = adult_dataset[adult_dataset$`native-country` != " ?",]


## Traversing tree using an instance from input data(x)
return_path <- function(x){
  node <- 1
  nspl <- 1
  path_list <- list()
  test_yval <- list()
  while(nspl != 0){
    i <- subset(x, select=-c(target))
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


## Local dt and paths list setup
final_paths <- data.frame(matrix(ncol = 1, nrow = nrow(adult_dataset)))

set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
names(set1) <- "new_col"
set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
set4 <- unite(data.frame(t(combn(LETTERS,4))), "new_col", sep = "")
set5 <- unite(data.frame(t(combn(LETTERS,5))), "new_col", sep = "")
bin_labels <- bind_rows(set1, set2, set3, set4, set5)

global_bins <- hash()

for (iter_index in 1:nrow(adult_dataset)) {
# for (iter_index in 1:1000) {
	print(iter_index)
    x <- adult_dataset[iter_index, 1:15]
    x <- x[rep(seq_len(nrow(x)), 250),]
    x <- x[rowSums(is.na(x)) == 0,]  
    x <- rbind(x, adult_dataset %>%
                 slice(-c(iter_index))%>%
                 slice(sample(nrow(adult_dataset)))%>%
                 slice(1:750))
    
    ## Create local tree and export details as csv
    local_tree <- rpart(target ~ ., data = x, minsplit=1,cp=0)
    frame_csv_name <- paste0("local_dt_info/frame", "_", iter_index,".csv")
    splits_csv_name <- paste0("local_dt_info/splits", "_", iter_index,".csv")
    write.csv(local_tree$frame, file=frame_csv_name)
    write.csv(local_tree$splits, file=splits_csv_name)
    
    temp_frame <- local_tree$frame

    nc <- temp_frame[, c("ncompete", "nsurrogate")]

    temp_frame$index <- 1L + c(0L, cumsum((temp_frame$var != "<leaf>") + nc[[1L]] + nc[[2L]]))[-(nrow(temp_frame) + 1L)] ## Validate the values!

    temp_frame$index[temp_frame$var == "<leaf>"] <- 0L

    nodes <- temp_frame[, c("n", "ncompete", "nsurrogate", "index")]

    nnum = row.names(temp_frame)

    vnum <- match(rownames(local_tree$splits), colnames(adult_dataset[1,]))

    split <- local_tree$splits

    split_rownames <- rownames(split)

    split_df <- as.data.frame(split, row.names = 0)

    csplit <- local_tree$csplit -2L

    path_list <- return_path(adult_dataset[iter_index,1:15])
    final_paths[iter_index, ] <- path_list
# write.csv(final_paths, file="test_adult_paths_1000.csv")
# label_df <- data.frame(values(global_bins))
# write.csv(label_df, file="test_adult_bin_labels_1000.csv")
}

## Export paths and bin details
write.csv(final_paths, file="test_adult_paths_36000.csv")
label_df <- data.frame(values(global_bins))
write.csv(label_df, file="test_adult_bin_labels_36000.csv")