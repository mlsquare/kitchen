import pandas as pd
import re
import pickle
import json
import argparse

def main(dataset):
    with open('../Configs/'+dataset+'.json') as config_file:
            config = json.load(config_file)

    paths = pd.read_csv('../Outputs/'+config['primary_paths'], header = 0)
    paths = paths.values
    bins = pd.read_csv('../Outputs/'+config['local_bins'], header = 0)
    bin_vals = bins.values

    bin_dict = dict((x[0], x[1]) for x in bin_vals)

    regex = re.compile(config['path_regex'], re.I)
        
    path_list = []

    for path in paths:
        nodes = path[0].split(",")
        newpath = []
        for node in nodes:
            matchobj =  re.match(regex, node)
            newpath.append((matchobj.group(1), bin_dict[matchobj.group(2)], matchobj.group(3)))
        path_list.append(newpath)

    pickle.dump(path_list, open('../Outputs/'+config['name']+'_path_list.dump', 'wb'))
    #pl = pickle.load(open('../Outputs/'+config['name']+'path_list.dump', 'rb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Instance metrics', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, required=True, help = 'name of the dataset')
    (args, _) = parser.parse_known_args()

    main(args.dataset)
