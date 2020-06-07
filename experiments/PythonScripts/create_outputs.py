import pandas as pd
import numpy as np
import json
import argparse

def tree_widths(dataset):
    with open('../Configs/'+dataset+'.json') as config_file:
        config = json.load(config_file)

    filename = '../R/'+config['frame_folder']+'/frame_'
    ext = '.csv'

    max_widths = []

    for i in range(1,config['sample']+1): # RANGE
        filepath = filename+str(i)+ext
        frame = pd.read_csv(filepath, header = 0)
        frame = frame.values
        nodes = frame[:, 0]
        nodes = np.sort(nodes)
        max_node = nodes[-1]
        start, end = 2, 4
        max_width, depth = 1, 1
        i = 1
        while start <= max_node:
            num = ((start <= nodes) & (nodes < end)).sum()
            if num>max_width:
                max_width, depth = num, i
            start, end = end, end*2
            i += 1
        max_widths.append([max_width, depth])

    pd.DataFrame(max_widths).to_csv("../Outputs/"+config['tree_widths'], header = ['width','at-depth'], index = False)

def leaf_nodes(dataset):
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

def main(dataset):
    tree_widths(dataset)
    leaf_nodes(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Instance metrics', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, required=True, help = 'name of the dataset')
    (args, _) = parser.parse_known_args()

    main(args.dataset)