import numpy as np
import pandas as pd
import json
import argparse

def convert_factors_to_numeric(dataset):

    # DATASET
    with open('../Configs/'+dataset+'.json') as config_file:
        config = json.load(config_file)
 
    dataset = pd.read_csv('../Data/'+config['filtered_data_with_headers'], header = 0) 
    factors = pd.read_csv('../Outputs/'+config['factors'], header = 0)

    dataset = dataset.values
    X = dataset

    factors = factors.values

    cols = factors[0,:].astype(int)

    k = 0

    for i in cols:
        col = X[:,i-1]
        for j, val in enumerate(factors[1:,k]):
            col[col == val] = j+1
        X[:, i-1] = col
        k += 1

    op = pd.DataFrame(X, columns=config['columns'])

    pd.DataFrame(op).to_csv("../Data/"+config['data_numeric'], index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Instance metrics', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, required=True, help = 'name of the dataset')
    (args, _) = parser.parse_known_args()

    convert_factors_to_numeric(args.dataset)