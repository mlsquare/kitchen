import pandas as pd
import numpy as np
import json
import argparse

def main(dataset):
    with open('../Configs/'+dataset+'.json') as config_file:
            config = json.load(config_file)

    filename = '../R/'+config['frame_folder']+'/frame_'
    ext = '.csv'

    leaves = []

    for i in range(1,config['sample']+1): # RANGE
        filepath = filename+str(i)+ext
        frame = pd.read_csv(filepath, header = 0)
        frame = frame.values
        nodes = frame[:, 0]
        nodes = np.sort(nodes)
        new_list = []
        for i in range(len(nodes)):
            if 2*nodes[i] in nodes[i:] or 2*nodes[i]+1 in nodes[i:]:
                continue
            new_list.append(nodes[i])
        leaves.append(new_list)

    num_leaves = [len(x) for x in leaves]

    op = pd.DataFrame(
        {'num_leaves': num_leaves,
        'leaves': leaves
        })

    pd.DataFrame(op).to_csv("../Outputs/"+config['leaf_nodes'], index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Instance metrics', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, required=True, help = 'name of the dataset')
    (args, _) = parser.parse_known_args()

    main(args.dataset)
