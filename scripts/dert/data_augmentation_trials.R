
library(tidyr)
library(dplyr)
library(hash)
library(rpart)
library(rpart.plot)
library(iterators)
library(foreach)
# library(class)
library(FNN)
data(ptitanic)
library(gridExtra)
library(grid)

## Load dataset
adult_dataset <- read.csv("../../data/raw/adult.data.csv", header = FALSE)
col_names <- c('age', 'workclass', 'fnlwgt', 'education', 'education-num',
         'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'target')
colnames(adult_dataset) <- col_names

adult_dataset = adult_dataset[adult_dataset["workclass"] != " ?",]
adult_dataset = adult_dataset[adult_dataset["occupation"] != " ?",]
adult_dataset = adult_dataset[adult_dataset$`native-country` != " ?",]

cont_var_names <- c('age','fnlwgt', 'education-num', 'capital-gain',
  'capital-loss', 'hours-per-week')

adult_dataset = adult_dataset[1:1000,]

scaled_data <- scale(adult_dataset[cont_var_names])
remaining_data <- adult_dataset[!(names(adult_dataset) %in% cont_var_names)]

adult_dataset <- merge(scaled_data, remaining_data, by.x=0, by.y=0)
adult_dataset <- adult_dataset[,-c(1)] # Remove "Row.name"


print(summary(adult_dataset))
# adult_dataset['age'] <- scale(adult_dataset['age'])
# adult_dataset['fnlwgt'] <- scale(adult_dataset['fnlwgt'])
# adult_dataset['education-num'] <- scale(adult_dataset['education-num'])
# adult_dataset['capital-gain'] <- scale(adult_dataset['capital-gain'])
# adult_dataset['capital-loss'] <- scale(adult_dataset['capital-loss'])
# adult_dataset['hours-per-week'] <- scale(adult_dataset['hours-per-week'])

# data(iris)


## Traversing tree using an instance from input data(x)
# return_path <- function(x){
#   node <- 1
#   nspl <- 1
#   path_list <- list()
#   test_yval <- list()
#   while(nspl != 0){
#     i <- subset(x, select=-c(Species))
#     npos <- match(node, nnum)
#     nspl <- nodes[4][npos,] # Recheck
#     var <- vnum[nspl]
#     ncat <- split_df[2][nspl,]
#     temp <- split_df[4][nspl,]
#       if (nspl > 0){
#         if (ncat >= 2){
#           dir = csplit[temp,as.integer(i[var])]
#           label <- ifelse(dir == -1, 0, 1)
#           level <- paste("(", as.character(i[var]), ")", sep = "")
#           if (!(level %in% keys(invert(global_bins)))){
#                 .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=level)
#           }
          
#           node_decision <- paste(as.character(var), invert(global_bins)[[level]], label, sep = "")
#           path_list <- c(path_list, node_decision)
#         }
#         else if (i[var] < temp){
#           if (!(as.character(temp) %in% keys(invert(global_bins)))){
#                 .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=as.character(temp))
#           }
#           label <- invert(global_bins)[[as.character(temp)]]
#           node_decision <- paste(as.character(var), label, 0, sep = "")
#           path_list <- c(path_list, node_decision)
#           dir = ncat
#         }
#         else {
#           if (!(as.character(temp) %in% keys(invert(global_bins)))){
#                 .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=as.character(temp))
#           }
#           label <- invert(global_bins)[[as.character(temp)]]
#           node_decision <- paste(as.character(var), label, 1, sep = "")
#           path_list <- c(path_list, node_decision)
#           dir = -ncat
#         }

#         if (dir == -1){
#           node = 2 * node
#         }
#         else{
#           node = 2 * node + 1
#         }
#       }
#     else{
#     }
#   }
#   path_list = paste(path_list, collapse = ",")
#   names(path_list) <- "new_col"
#   return(path_list)
# }


# ## Local dt and paths list setup
# final_paths <- data.frame(matrix(ncol = 2, nrow = 1500))
# colnames(final_paths) <- c("index", "paths")

# set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
# names(set1) <- "new_col"
# set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
# set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
# set4 <- unite(data.frame(t(combn(LETTERS,4))), "new_col", sep = "")
# set5 <- unite(data.frame(t(combn(LETTERS,5))), "new_col", sep = "")
# bin_labels <- bind_rows(set1, set2, set3, set4, set5)

# global_bins <- hash()
# j = 0
# for (iter_index in 1:nrow(iris)) {
#   j = j + 1
# # for (iter_index in 1:1000) {
# 	print(iter_index)
#     x <- iris[iter_index, 1:5]
#     x <- x[rep(seq_len(nrow(x)), 10),]
#     x <- x[rowSums(is.na(x)) == 0,]  
#     # x <- rbind(x, iris[,1:4] %>%
#     #              slice(-c(iter_index))%>%
#     #              slice(sample(nrow(iris)))%>%
#     #              slice(1:10))
#     x <- rbind(x,iris[sample(nrow(iris))[1:10],1:5])
    
#     ## Create local tree and export details as csv
#     local_tree <- rpart(Species ~ ., data = x, minsplit=1,cp=0)
#     frame_csv_name <- paste0("local_dt_info/iris_temp_frame", "_", iter_index,".csv")
#     splits_csv_name <- paste0("local_dt_info/iris_temp_splits", "_", iter_index,".csv")
#     write.csv(local_tree$frame, file=frame_csv_name)
#     write.csv(local_tree$splits, file=splits_csv_name)
    
#     temp_frame <- local_tree$frame

#     nc <- temp_frame[, c("ncompete", "nsurrogate")]

#     temp_frame$index <- 1L + c(0L, cumsum((temp_frame$var != "<leaf>") + nc[[1L]] + nc[[2L]]))[-(nrow(temp_frame) + 1L)] ## Validate the values!

#     temp_frame$index[temp_frame$var == "<leaf>"] <- 0L

#     nodes <- temp_frame[, c("n", "ncompete", "nsurrogate", "index")]

#     nnum = row.names(temp_frame)

#     vnum <- match(rownames(local_tree$splits), colnames(iris[1,]))

#     split <- local_tree$splits

#     split_rownames <- rownames(split)

#     split_df <- as.data.frame(split, row.names = 0)

#     csplit <- local_tree$csplit -2L

#     path_list <- return_path(iris[iter_index,])

#     final_paths[j, "paths"] <- path_list
#     final_paths[j, "index"] <- iter_index
#     for (i in 11:20){
#       j = j+1
#       path_list <- return_path(iris[i,])
#       index <- as.numeric(rownames(x[i,]))
#       final_paths[j, "paths"] <- path_list
#       final_paths[j, "index"] <- index

#     }
# # write.csv(final_paths, file="test_adult_paths_1000.csv")
# # label_df <- data.frame(values(global_bins))
# # write.csv(label_df, file="test_adult_bin_labels_1000.csv")
# }




# ## Export paths and bin details
# write.csv(final_paths, file="test_iris_paths_aug.csv")
# label_df <- data.frame(values(global_bins))
# write.csv(label_df, file="test_iris_bin_labels_aug.csv")


##### augmentation workflow #####

# return_path <- function(x){
#   node <- 1
#   nspl <- 1
#   path_list <- list()
#   test_yval <- list()
#   while(nspl != 0){
#     i <- subset(x, select=-c(target))
#     npos <- match(node, nnum)
#     nspl <- nodes[4][npos,] # Recheck
#     var <- vnum[nspl]
#     ncat <- split_df[2][nspl,]
#     temp <- split_df[4][nspl,]
#       if (nspl > 0){
#         if (ncat >= 2){
#           dir = csplit[temp,as.integer(i[var])]
#           label <- ifelse(dir == -1, 0, 1)
#           level <- paste("(", as.character(i[var]), ")", sep = "")
#           if (!(level %in% keys(invert(global_bins)))){
#                 .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=level)
#           }
          
#           node_decision <- paste(as.character(var), invert(global_bins)[[level]], label, sep = "")
#           path_list <- c(path_list, node_decision)
#         }
#         else if (i[var] < temp){
#           if (!(as.character(temp) %in% keys(invert(global_bins)))){
#                 .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=as.character(temp))
#           }
#           label <- invert(global_bins)[[as.character(temp)]]
#           node_decision <- paste(as.character(var), label, 0, sep = "")
#           path_list <- c(path_list, node_decision)
#           dir = ncat
#         }
#         else {
#           if (!(as.character(temp) %in% keys(invert(global_bins)))){
#                 .set(global_bins, keys=bin_labels$new_col[length(global_bins)+1], values=as.character(temp))
#           }
#           label <- invert(global_bins)[[as.character(temp)]]
#           node_decision <- paste(as.character(var), label, 1, sep = "")
#           path_list <- c(path_list, node_decision)
#           dir = -ncat
#         }

#         if (dir == -1){
#           node = 2 * node
#         }
#         else{
#           node = 2 * node + 1
#         }
#       }
#     else{
#     }
#   }
#   path_list = paste(path_list, collapse = ",")
#   names(path_list) <- "new_col"
#   return(path_list)
# }


# ## Local dt and paths list setup
# final_paths <- data.frame(matrix(ncol = 2, nrow = nrow(adult_dataset)))
# # colnames(final_paths) <- c("paths")
# colnames(final_paths) <- c("index", "paths")

# set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
# names(set1) <- "new_col"
# set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
# set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
# set4 <- unite(data.frame(t(combn(LETTERS,4))), "new_col", sep = "")
# set5 <- unite(data.frame(t(combn(LETTERS,5))), "new_col", sep = "")
# bin_labels <- bind_rows(set1, set2, set3, set4, set5)

# # global_tree <- rpart(Species ~ ., data = iris, minsplit=1,cp=0)

# global_bins <- hash()
# j = 0
# # for (iter_index in 1:nrow(iris)) {
#   j = j + 1
# for (iter_index in 1:1000) {
#   print(iter_index)
#     x <- adult_dataset[iter_index, 1:15]
#     x <- x[rep(seq_len(nrow(x)), 50),]
#     x <- x[rowSums(is.na(x)) == 0,]  
#     # x <- rbind(x, iris[,1:4] %>%
#     #              slice(-c(iter_index))%>%
#     #              slice(sample(nrow(iris)))%>%
#     #              slice(1:10))
#     x <- rbind(x,adult_dataset[sample(nrow(adult_dataset))[1:50],1:15])
    
#     # Create local tree and export details as csv
#     local_tree <- rpart(target ~ ., data = x, minsplit=1,cp=0)
#     frame_csv_name <- paste0("local_dt_info/adult_temp_frame", "_", iter_index,".csv")
#     splits_csv_name <- paste0("local_dt_info/adult_temp_splits", "_", iter_index,".csv")
#     write.csv(local_tree$frame, file=frame_csv_name)
#     write.csv(local_tree$splits, file=splits_csv_name)
    
#     temp_frame <- local_tree$frame

#     nc <- temp_frame[, c("ncompete", "nsurrogate")]

#     temp_frame$index <- 1L + c(0L, cumsum((temp_frame$var != "<leaf>") + nc[[1L]] + nc[[2L]]))[-(nrow(temp_frame) + 1L)] ## Validate the values!

#     temp_frame$index[temp_frame$var == "<leaf>"] <- 0L

#     nodes <- temp_frame[, c("n", "ncompete", "nsurrogate", "index")]

#     nnum = row.names(temp_frame)

#     vnum <- match(rownames(local_tree$splits), colnames(adult_dataset[1,]))

#     split <- local_tree$splits

#     split_rownames <- rownames(split)

#     split_df <- as.data.frame(split, row.names = 0)

#     csplit <- local_tree$csplit -2L

#     # print(adult_dataset[iter_index,])
#     path_list <- return_path(adult_dataset[iter_index,])
#     # print(path_list)

#     final_paths[iter_index, "paths"] <- path_list
#     final_paths[j, "index"] <- iter_index
#     for (i in 51:100){
#       j = j+1
#       path_list <- return_path(x[i,])
#       index <- as.numeric(rownames(x[i,]))
#       final_paths[j, "paths"] <- path_list
#       final_paths[j, "index"] <- index

#     }
# # write.csv(final_paths, file="test_adult_paths_1000.csv")
# # label_df <- data.frame(values(global_bins))
# # write.csv(label_df, file="test_adult_bin_labels_1000.csv")
# }




# ## Export paths and bin details
# write.csv(final_paths, file="local_adult_aug_paths.csv")
# label_df <- data.frame(values(global_bins))
# write.csv(label_df, file="local_adult_aug_bin_labels.csv")



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
    print(split_df)
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

global_tree <- rpart(target ~ ., data = adult_dataset)
## Local dt and paths list setup
final_paths <- data.frame(matrix(ncol = 1, nrow = nrow(adult_dataset)))
# colnames(final_paths) <- c("paths")
colnames(final_paths) <- c("paths")

set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
names(set1) <- "new_col"
set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
set4 <- unite(data.frame(t(combn(LETTERS,4))), "new_col", sep = "")
set5 <- unite(data.frame(t(combn(LETTERS,5))), "new_col", sep = "")
bin_labels <- bind_rows(set1, set2, set3, set4, set5)

# global_tree <- rpart(Species ~ ., data = iris, minsplit=1,cp=0)
# itx <- iter(iris[,1:5], by = "row")

pdf("test_plot_adult_dataset_1.pdf") #Outside the loop

path.to.root <- function(node){
  if(node == 1)   # root?
    node
  else            # recurse, %/% 2 gives the parent of node
    c(node, path.to.root(node %/% 2))
  }
global_bins <- hash()
j = 0
# for (iter_index in 1:nrow(iris)) {
  j = j + 1
for (iter_index in 1:100) {
  print(iter_index)
    x <- adult_dataset[iter_index, 1:15]
    x <- x[rep(seq_len(nrow(x)), 50),]
    x <- x[rowSums(is.na(x)) == 0,]  # Cross check the reason!
    # x <- rbind(x, iris[,1:4] %>%
    #              slice(-c(iter_index))%>%
    #              slice(sample(nrow(iris)))%>%
    #              slice(1:10))
    x <- rbind(x,adult_dataset[sample(nrow(adult_dataset))[1:50],1:15]) ## 
    print(dim(x))
    
    # Create local tree and export details as csv
    local_tree <- rpart(target ~ ., data = x, minsplit=1,cp=0)
    frame_csv_name <- paste0("local_dt_info/adult_temp_frame", "_", iter_index,".csv")
    splits_csv_name <- paste0("local_dt_info/adult_temp_splits", "_", iter_index,".csv")
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

    # print(adult_dataset[iter_index,])
    path_list <- return_path(adult_dataset[iter_index,])
    # print(path_list)
    print(path_list)

    final_paths[iter_index, "paths"] <- path_list

    #### plot results -- START ####

    node <- as.integer(rpart.predict(global_tree, adult_dataset[iter_index,], nn = TRUE)['nn'])
    nodes <- as.numeric(row.names(global_tree$frame))
    global_cols <- ifelse(nodes %in% path.to.root(node), "sienna", "gray")

    node <- as.integer(rpart.predict(local_tree, adult_dataset[iter_index,], nn = TRUE)['nn'])
    nodes <- as.numeric(row.names(local_tree$frame))
    local_cols <- ifelse(nodes %in% path.to.root(node), "sienna", "gray")

    # layout(matrix(c(1,2,3,3,4,4,4),nrow = 3, byrow=TRUE))
    # layout(matrix(c(1,2,3,3),nrow = 2, byrow=TRUE))
    layout(matrix(c(1,2,3,3,4,4),3,2, byrow=TRUE))
    test_plot = prp(global_tree, nn=TRUE, col=global_cols, branch.col=global_cols, split.col=global_cols, nn.col=global_cols, compress = FALSE, extra=109)       #Global tree
    test_plot = prp(local_tree, nn=TRUE, col=local_cols, branch.col=local_cols, split.col=local_cols, nn.col=local_cols, compress = FALSE, extra=109)           #Local tree
    table_1 <- grid.table(adult_dataset[iter_index,])
    grid.newpage()
    table_2 <- grid.table(t(as.data.frame(table(x[, 5]))))
    # grid.arrange(table_1, table_2, nrow=2)

    #### Plot results -- END ####
    # final_paths[j, "index"] <- iter_index
    # for (i in 51:100){
    #   j = j+1
    #   path_list <- return_path(x[i,])
    #   index <- as.numeric(rownames(x[i,]))
    #   final_paths[j, "paths"] <- path_list
    #   final_paths[j, "index"] <- index

    # }
# write.csv(final_paths, file="test_adult_paths_1000.csv")
# label_df <- data.frame(values(global_bins))
# write.csv(label_df, file="test_adult_bin_labels_1000.csv")
}

# itx <- iter(iris[,1:5], by = "row")

# pdf("test_plot.pdf") #Outside the loop

# path.to.root <- function(node){
#   if(node == 1)   # root?
#     node
#   else            # recurse, %/% 2 gives the parent of node
#     c(node, path.to.root(node %/% 2))
#   }

# # foreach(i = itx, j=icount(), .combine = "c") %dopar% {
# for (iter_index in 1:dim(adult_dataset)[1]) {
# # for (iter_index in 1){
#     # x <- data.frame(i)
#     # x <- data.frame(iris[iter_index,1:5])
#     x <- adult_dataset[iter_index, 1:15]
#     x <- x[rep(seq_len(nrow(x)), 50),]
#     x <- x[rowSums(is.na(x)) == 0,]
#     remaining_df <- adult_dataset[-c(iter_index),] # exclude current instance!
#     random_sample <- remaining_df[sample(nrow(adult_dataset)),]
#     x <- rbind(x, random_sample[1:50,])
#     # print(x)
#     ctrl <- rpart.control(maxdepth=2)
#     local_tree <- rpart(Species ~ ., data = x, minsplit=1,cp=0 ,control=ctrl)
    
#     #### plot results -- START ####

#     node <- as.integer(rpart.predict(global_tree, iris[iter_index,], nn = TRUE)['nn'])
#     nodes <- as.numeric(row.names(global_tree$frame))
#     global_cols <- ifelse(nodes %in% path.to.root(node), "sienna", "gray")

#     node <- as.integer(rpart.predict(local_tree, iris[iter_index,], nn = TRUE)['nn'])
#     nodes <- as.numeric(row.names(local_tree$frame))
#     local_cols <- ifelse(nodes %in% path.to.root(node), "sienna", "gray")

#     # layout(matrix(c(1,2,3,3,4,4,4),nrow = 3, byrow=TRUE))
#     # layout(matrix(c(1,2,3,3),nrow = 2, byrow=TRUE))
#     layout(matrix(c(1,2,3,3,4,4),3,2, byrow=TRUE))
#     test_plot = prp(global_tree, nn=TRUE, col=global_cols, branch.col=global_cols, split.col=global_cols, nn.col=global_cols, compress = FALSE, extra=109)       #Global tree
#     test_plot = prp(local_tree, nn=TRUE, col=local_cols, branch.col=local_cols, split.col=local_cols, nn.col=local_cols, compress = FALSE, extra=109)           #Local tree
#     table_1 <- grid.table(iris[iter_index,])
#     grid.newpage()
#     table_2 <- grid.table(t(as.data.frame(table(x[, 5]))))
#     # grid.arrange(table_1, table_2, nrow=2)

#     #### Plot results -- END ####
# }

dev.off() #Outside the loop




## Export paths and bin details
write.csv(final_paths, file="local_adult_norm_path.csv")
label_df <- data.frame(values(global_bins))
write.csv(label_df, file="local_adult_norm_bins.csv")
write.csv(adult_dataset, file="normalized_adult_dataset.csv")