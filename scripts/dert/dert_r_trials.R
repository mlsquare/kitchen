```{r}
install.packages('rpart')
install.packages('rpart.plot')
install.packages("tidyr")
install.packages('dplyr')
install.packages('foreach')
# install.packages('class')# KNN
install.packages('FNN') # KNN
install.packages('iterators')
install.packages('foreach')
```


```{r Load packages}
## Load packages
library(rpart)
library(rpart.plot)
library(tidyr)
library(dplyr)
library(iterators)
library(foreach)
# library(class)
library(FNN)
data(ptitanic)

str(ptitanic)
```


```{r}
# Load iris
data(iris)
```

```{r}
# Tree for iris dataset

set.seed(123)
# tree <- rpart(Species ~ ., data = iris, control = rpart.control(cp = 0.0001))
tree <- rpart(Species ~ ., data = iris, minsplit=1,cp=0)

printcp(tree)
```

```{r return_path function}
## Traversing tree using an instance from input data(x)
return_path <- function(x){
  node <- 1
  nspl <- 1
  path_list <- list()
  test_yval <- list()
  # temp_path <- list()
  while(nspl != 0){
    # i <- x
    # i <- subset(x, select=-c(survived))
    i <- subset(x, select=-c(Species))
    npos <- match(node, nnum)
    nspl <- nodes[4][npos,] # Recheck
    var <- vnum[nspl]
    ncat <- split_df[2][nspl,]
    temp <- split_df[4][nspl,]
      if (nspl > 0){
        if (ncat >= 2){
          dir = csplit[temp,as.integer(i[var])]
          # levels <- bin_labels[[1]][length(bin_list)+1]
          # names(levels) <- as.integer(x[var])
          # path_list <- c(path_list, bin_list[[1]][-1])
          label <- ifelse(dir == -1, 0, 1)
          level <- paste("(", as.character(i[var]), ")", sep = "")
          node_decision <- paste(as.character(var), bin_list[[level]], label, sep = "")
          # path_list <- c(path_list, var,bin_list[[level]])
          path_list <- c(path_list, node_decision)
          # print(level)
          # print(dir)
        }
        else if (i[var] < temp){
          print(temp)
          # label <- ifelse(ncat==-1, bin_list[[as.character(temp)]][1], bin_list[[as.character(temp)]][2]) # Validate results!
          label <- bin_list[[as.character(temp)]][1]
          print(label)
          node_decision <- paste(as.character(var), label, 0, sep = "")
          # print(label)
          # path_list <- c(path_list, var,label)
          path_list <- c(path_list, node_decision)
          dir = ncat
          # print(dir)
        }
        else {
          # label <- ifelse(ncat==1, bin_list[[as.character(temp)]][1], bin_list[[as.character(temp)]][2]) # Validate results!
          label <- bin_list[[as.character(temp)]][1]
          node_decision <- paste(as.character(var), label, 1, sep = "")
          # print(label)
          # path_list <- c(path_list, var,label)
          path_list <- c(path_list, node_decision)
          dir = -ncat
          # print(dir)
        }

        if (dir == -1){
          node = 2 * node
        }
        else{
          node = 2 * node + 1
        }
      }
    else{
      # print("leaf node")
      # test_yval <- c(test_yval, tree$frame$yval[node])
    }
  }
  path_list = paste(path_list, collapse = ",")
  names(path_list) <- "new_col"
  return(path_list)
}
```

```{r}
get_updated_binlist <- function(new_boundaries, bin_list){
  new_boundaries <- new_boundaries[!(new_boundaries[[1]] %in% names(bin_list)),]
  if (length(new_boundaries)>0){
    for (i in 1: length(new_boundaries)){
    index <- length(bin_list) + 1
    # print("index -- ")
    # print(index)
    val <- list(list(bin_labels[[1]][index]))
    # print("val")
    # print(val)
    # print("i")
    # print(i)
    names(val) <- as.character(new_boundaries[i])
    bin_list <- c(bin_list, val)
    }
  }
  return(bin_list)
}
```



```{r Local tree trials}
itx <- iter(iris[,1:5], by = "row")
# itx <- iter(k, by = "row")
# itx <- iter(index_maps[1:5,], by = "row")
final_paths <- data.frame(matrix(ncol = 1, nrow = nrow(iris)))
# colnames(final_paths) <- c("new_col")

set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
names(set1) <- "new_col"
set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
bin_labels <- bind_rows(set1, set2, set3)

# bin_list <- list()

# label_list <- list()

bin_tracker <- 1
label_tracker <- 1
bin_list <- list()
bin_df <- data.frame()


## changes to fix memory issues
# 1) Move bin creation to a function - bin_df, bin_list. Check for duplicates and return updated list.
# 2) Move bin_labels out of loops


set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
names(set1) <- "new_col"
set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
bin_labels <- bind_rows(set1, set2, set3)

# foreach(i = itx, j=icount(), .combine = "c") %dopar% {
for (iter_index in 1:dim(iris)[1]) {
    # x <- data.frame(i)
    x <- data.frame(iris[iter_index,1:5])
    # print(i)
    # itx_2 <- iter(i, by = "column")
    # foreach(a=itx_2, b=icount(), .combine = "c") %do% {
    #   x <- rbind(x, random_sample[a,])
    # }
    # for (a in sample(10)){
    #   print('loop entered')
    #   x <- rbind(x, random_sample[a,])
    # }
    random_sample <- iris[sample(nrow(iris)),] # exclude current instance!
    x <- rbind(x, random_sample[1:9,])
    # print(x)
    local_tree <- rpart(Species ~ ., data = x, minsplit=1,cp=0)
    # print(local_tree)
    # For local trees
    cat_var = c("sex", "pclass")
    bin_boundaries <- local_tree$splits[!names(local_tree$splits[,4]) %in% cat_var,4]
    bin_df <- data.frame(sort(unique(bin_boundaries)))
    # temp_df <- data.frame(sort(unique(bin_boundaries)))
    # colnames(temp_df) <- "new_col"
    # bin_df <- rbind(bin_df, temp_df)
    # names(bin_df) <- "new_col"
    # bin_df <- rbind(bin_df, cat_bind) # for cat_var

    # set1 <- data.frame(LETTERS, stringsAsFactors=FALSE)
    # names(set1) <- "new_col"
    # set2 <- unite(data.frame(t(combn(LETTERS,2))), "new_col", sep = "")
    # set3 <- unite(data.frame(t(combn(LETTERS,3))), "new_col", sep = "")
    # bin_labels <- bind_rows(set1, set2, set3)

    # bin_df = transform(bin_df, new_col = as.character(new_col))
    names(bin_df) <- "new_col"

    # bin_list <- list()
    # 
    # itx_3 <- iter(bin_df[[1]], by = "row")
    # ## Append
    # # foreach(i = itx_3, k=icount(), .combine = "c") %dopar% {
    # for (k in 1:length(bin_df[[1]])){
    #   if ("(" == strsplit(bin_df[[1]][k], split="")[[1]][1]){
    #     # index <- dim(bin_df)[1]
    #     val <- bin_labels[[1]][bin_tracker]
    #     }
    #   else{
    #     # index <- dim(bin_df)[1]
    #     val <- list(list(bin_labels[[1]][bin_tracker]))
    #   }
    #   cutpoint <- bin_df[[1]][k]
    #   # names(val) <- as.character(i)
    #   names(val) <- as.character(cutpoint)
    #   bin_tracker = bin_tracker + 1
    #   print("i -- ")
    #   print(i)
    #   print("bin_list")
    #   print(names(bin_list))
    #   # if (i %in% names(bin_list)){
    #   if (bin_df[[1]][k] %in% names(bin_list)){
    #     print("return NULL -- ")
    #     }
    #   else{
    #     bin_list <- c(bin_list, val)
    #   }
    # }
    
    bin_list <- get_updated_binlist(bin_df, bin_list)

    #label_list <- list()

    #itx_4 <- iter(bin_df[[1]], by = "row")

    #label_list <- foreach(i = itx_4, l=icount(), .combine = "c") %do% {
    #  val <- i
    #  names(val) <- bin_labels[[1]][l]
    #  val
    #}
    temp_frame <- local_tree$frame

    nc <- temp_frame[, c("ncompete", "nsurrogate")]

    temp_frame$index <- 1L + c(0L, cumsum((temp_frame$var != "<leaf>") + nc[[1L]] + nc[[2L]]))[-(nrow(temp_frame) + 1L)] ## Validate the values!

    temp_frame$index[temp_frame$var == "<leaf>"] <- 0L

    nodes <- temp_frame[, c("n", "ncompete", "nsurrogate", "index")]

    nnum = row.names(temp_frame)

    # vnum <- match(rownames(tree$splits), colnames(x))
    vnum <- match(rownames(local_tree$splits), colnames(iris[1,]))

    split <- local_tree$splits

    split_rownames <- rownames(split)

    split_df <- as.data.frame(split, row.names = 0)

    csplit <- local_tree$csplit -2L

    path_list <- return_path(iris[iter_index,1:5])
    # print(path_list)
    # print(j)
    final_paths[iter_index, ] <- path_list
    # cutpoints[j, ] <- path_list
    }
```


```{r}
write.csv(final_paths, file="paths.csv")
```

```{r}

itx_3 <- iter(bin_df[[1]], by = "row")
## Append
bin_list <- list()
bin_tracker <- 1
foreach(i = itx_3, k=icount()) %do% {
  if ("(" == strsplit(bin_df[[1]][k], split="")[[1]][1]){
    # index <- dim(bin_df)[1]
    val <- bin_labels[[1]][bin_tracker]
    }
  else{
    index <- dim(bin_df)[1]
    val <- list(list(bin_labels[[1]][bin_tracker]))
    }
  names(val) <- as.character(i)
  # bin_tracker = bin_tracker + 1
  # print("i -- ")
  # print(i)
  # print("bin_list")
  bin_tracker <- bin_tracker + 1
  # print(names(bin_list))
  if (i %in% names(bin_list)){
    print("return NULL -- ")
    # return(NULL)
    }
  else{
    bin_list = c(bin_list, val)
  }
}

```

```{r}
for (i in 1: length(bin_df[[1]])){
  print(bin_df[[1]][i])
}
```


