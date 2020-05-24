import numpy as np
import pandas as pd
import json
import re
from statistics import mean
import argparse

class InstanceMetrics:

    def __init__(self, inst, instance, decision_paths, weights, label, features, depths):
        self.inst = inst
        self.instance = instance
        self.decision_paths = decision_paths
        self.weights = weights
        self.label = label
        self.features =  features
        self.depths = depths
        self.co_occurrence_matrix = np.zeros((len(features), len(features)))
        self.path_lengths = []
        self.overlap_cache = dict()

        for path in decision_paths:
            self.path_lengths.append(len(path))

    # Co occurrence matrix for features

    def co_occurrence(self):
        for path in self.decision_paths:
            for i, pred1 in enumerate(path):
                for j, pred2 in enumerate(path):
                    if i>=j:
                        continue
                    q = min(pred1[0], pred2[0])
                    p = max(pred1[0], pred2[0])
                    if p!=q:
                        self.co_occurrence_matrix[p-1][q-1] += 1
    
    # Average length of paths for the instance
    
    def average_path_length(self):
        avg = 0
        for i in range(len(self.path_lengths)):
            avg += self.path_lengths[i] * self.weights[i]
        avg /= sum(self.weights)
        return avg

    # Average frequency of each feature in all paths at all depths for the instance

    def feature_frequency(self):
        freq = np.zeros(len(self.features))
        for i, path in enumerate(self.decision_paths):
            for pred in path:
                freq[pred[0]-1] += self.weights[i]
        freq /= freq.sum(axis=0, keepdims=True)
        #freq = np.true_divide(freq, freq.sum(axis=0, keepdims=True))
        return freq

    # (Rooted at k) Frequency of all features occurring at a depth in all paths for the instance 
    # Returns an array of frequencies of features at depth d

    def frequency_at_depth(self, depth):
        freq = np.zeros(len(self.features))
        for i, path in enumerate(self.decision_paths):
            if len(path)>depth: # depth is valid
                freq[path[depth][0]-1] += self.weights[i]
        s = freq.sum(axis=0, keepdims=True)
        if s!=0:
            freq /= s
        else:
            freq = np.zeros(len(self.features))
        return freq

    # Frequency of all features occuring at all depths
    # Returns a matrix of frequencies of features at all depth

    def frequency_at_all_depths(self):
        freq = []
        for d in range(len(self.decision_paths)):
            freq_d = self.frequency_at_depth(d)
            if np.count_nonzero(freq_d)==0:
                break
            freq.append(freq_d)
        freq = np.vstack(freq)
        return freq

    def display(self):
        print("Metrics for instance", self.inst)
        print("Co-occurrence matrix")
        self.co_occurrence()
        print(self.co_occurrence_matrix)

        print("Average path length")
        print(self.average_path_length())

        print("Depth")
        print(self.depths[self.inst])

        print("Mean rank of each feature")
        print(self.feature_frequency())

        print("Frequency of each feature at all depths (RAK)")
        print(self.frequency_at_all_depths())


def main(dataset):

    # DATASET
    with open('../Configs/'+dataset+'.json') as config_file:
        config = json.load(config_file)

    primary_paths = pd.read_csv('../Outputs/'+config['primary_paths'], header = 0)
    secondary_paths = pd.read_csv('../Outputs/'+config['secondary_paths'], header = 0, index_col=0)
    dataset = pd.read_csv('../Data/'+config['data_with_headers'], header = 0)
    bins = pd.read_csv('../Outputs/'+config['local_bins'], header = 0)
    depths = pd.read_csv('../Outputs/'+config['tree_depths'], header = 0)

    # Set instance

    inst = 0

    # PATHS

    primary_paths = primary_paths.values
    secondary_paths = secondary_paths.values
    paths_i = secondary_paths[:,inst]
    paths_i = paths_i[:np.argwhere(pd.isnull(paths_i))[0][0]]
    np.insert(paths_i, inst, primary_paths[inst], axis=0)

    bin_vals = bins.values
    bin_dict = dict((x[0], x[1]) for x in bin_vals)

    regex = re.compile(config['path_regex'], re.I)

    path_list = []

    for path in paths_i:
        nodes = path.split(",")
        newpath = []
        for node in nodes:
            matchobj =  re.match(regex, node)
            newpath.append((int(matchobj.group(1)), bin_dict[matchobj.group(2)], matchobj.group(3)))
        path_list.append(newpath)

    # WEIGHTS

    weights = np.repeat(config['secondary_weight'], len(paths_i)-1)
    weights = np.insert(weights, 0, config['primary_weight'], axis=0)

    # DEPTHS

    depths = depths.values
    depths = depths.flatten()

    # DATA AND LABELS

    dataset = dataset.values[0:config['sample']]
    X = dataset[:,0:config['num_features']]
    labels = dataset[:,config['target_col']-1]

    # FEATURE INDEXES

    features = np.arange(1,config['num_features']+1)

    # MAIN

    metrics = InstanceMetrics(inst, X[inst,:], path_list, weights, labels[inst], features, depths)

    metrics.display()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Instance metrics', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, required=True, help = 'name of the dataset')
    (args, _) = parser.parse_known_args()

    main(args.dataset)



