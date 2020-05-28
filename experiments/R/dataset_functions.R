dataset_source_csv <- function(dataset_name){
  switch(dataset_name, 
         adult = {return("Data/adult_headers.csv")},
         auto = {return("Data/auto_headers.csv")}
         )
}

dataset_col_names <- function(dataset_name){
  switch(dataset_name,
         adult = {
           return(c('age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'target'))
           },
         auto = {
           return(c("cylinders","displacement","horsepower","weight","acceleration","model","origin","mpg"))
         }
         )
}

dataset_preprocessed <- function(dataset_name, dataset){
  if(dataset_name == "adult"){
    dataset = dataset[dataset["workclass"] != " ?",]
    dataset = dataset[dataset["occupation"] != " ?",]
    dataset = dataset[dataset$`native-country` != " ?",]
    return(dataset)
  }
  if(dataset_name == "auto"){
    dataset = dataset[dataset["horsepower"] != " ?",]
    return(dataset)
  }
}

get_ntrees <- function(dataset_name){
  if(dataset_name == "adult"){
    return(1000);
  }
  if(dataset_name == "auto"){
    return(392);
  }
}

get_start <- function(dataset_name){
  if(dataset_name == "adult"){
    return(1);
  }
  if(dataset_name == "auto"){
    return(1);
  }
}

get_nsample <- function(dataset_name){
  if(dataset_name == "adult"){
    return(100);
  }
  if(dataset_name == "auto"){
    return(100);
  }
}

get_nprim <- function(dataset_name){
  if(dataset_name == "adult"){
    return(25);
  }
  if(dataset_name == "auto"){
    return(25);
  }
}

get_nsec <- function(dataset_name){
  if(dataset_name == "adult"){
    return(75);
  }
  if(dataset_name == "auto"){
    return(75);
  }
}

