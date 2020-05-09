import numpy as np
import pandas as pd
import re
from statistics import mean

class InstanceMetrics:

    def __init__(self, inst, instance, decision_paths, weights, label, features):
        self.inst = inst
        self.instance = instance
        self.decision_paths = decision_paths
        self.weights = weights
        self.label = label
        self.features =  features
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
            avg += self.path_lengths[i] * weights[i]
        avg /= sum(weights)
        return avg

    # Average frequency of each feature in all paths at all depths for the instance

    def feature_frequency(self):
        freq = np.zeros(len(features))
        for i, path in enumerate(self.decision_paths):
            for pred in path:
                freq[pred[0]-1] += weights[i]
        freq /= freq.sum(axis=0, keepdims=True)
        #freq = np.true_divide(freq, freq.sum(axis=0, keepdims=True))
        return freq

    # (Rooted at k) Frequency of all features occurring at a depth in all paths for the instance 
    # Returns an array of frequencies of features at depth d

    def frequency_at_depth(self, depth):
        freq = np.zeros(len(features))
        for i, path in enumerate(self.decision_paths):
            if len(path)>depth: # depth is valid
                freq[path[depth][0]-1] += weights[i]
        s = freq.sum(axis=0, keepdims=True)
        if s!=0:
            freq /= s
        else:
            freq = np.zeros(len(features))
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
        print(depths[inst])

        print("Mean rank of each feature")
        print(self.feature_frequency())

        print("Frequency of each feature at all depths (RAK)")
        print(self.frequency_at_all_depths())

primary_paths = pd.read_csv('../Outputs/iris_scaled_local_paths_100.csv', header = 0)
secondary_paths = pd.read_csv('../Outputs/iris_scaled_local_paths_per_instance_100.csv', header = 0, index_col=0)
iris = pd.read_csv('../Data/iris_headers.csv', header = 0)
bins = pd.read_csv('../Outputs/iris_scaled_local_bin_labels_100.csv', header = 0)
depths = pd.read_csv('../Outputs/iris_scaled_local_depths_100.csv', header = 0)

# Set instance
inst = 0

# PATHS

primary_paths = primary_paths.values
secondary_paths = secondary_paths.values
paths_i = secondary_paths[:,inst]
paths_i = paths_i[:np.argwhere(pd.isnull(paths_i))[0][0]]
np.insert(paths_i, inst, primary_paths[inst], axis=0)
#print(paths_i)

bin_vals = bins.values
bin_dict = dict((x[0], float(x[1])) for x in bin_vals)

regex = re.compile('([1-4])([A-Z]+)([01])', re.I)

path_list = []

for path in paths_i:
    nodes = path.split(",")
    newpath = []
    for node in nodes:
        matchobj =  re.match(regex, node)
        newpath.append((int(matchobj.group(1)), bin_dict[matchobj.group(2)], matchobj.group(3)))
    path_list.append(newpath)

# WEIGHTS

weights = np.repeat(0.01, len(paths_i)-1)
weights = np.insert(weights, 0, 0.5, axis=0)

# DEPTHS

depths = depths.values
depths = depths.flatten()

# DATA AND LABELS

dataset = iris.values
X = dataset[:,0:4].astype(float)
labels = dataset[:,4]

# FEATURE INDEXES

features = [1,2,3,4]

# MAIN

metrics = InstanceMetrics(inst, X[inst,:], path_list, weights, labels[inst], features)

metrics.display()


