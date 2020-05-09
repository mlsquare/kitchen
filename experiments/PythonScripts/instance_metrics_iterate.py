import numpy as np
import pandas as pd
import re
from statistics import mean, variance, stdev
import seaborn as sns
import matplotlib.pyplot as plt

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

    # Number of distinct features of a path

    def distinct_features_per_path(self, path):
        features = [predicate[0] for predicate in path]
        distinct_features = set(features)
        return distinct_features

    # Average number of distinct features

    def avg_distinct_features(self):
        num = 0
        for i, path in enumerate(self.decision_paths):
            distfeat = self.distinct_features_per_path(path)
            num += len(distfeat) *weights[i]
        num /= sum(weights)
        return num
    
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
        print("Co-occurrence matrix")
        self.co_occurrence()
        print(self.co_occurrence_matrix)

        print("Average path length")
        print(self.average_path_length())

        print("Depth")
        print(depths[self.inst])

        print("Mean rank of each feature")
        print(self.feature_frequency())

        print("Frequency of each feature at all depths (RAK)")
        print(self.frequency_at_all_depths())
        print("\n")

        print("Ratio of depth to average path length")
        print(depths[inst]/self.average_path_length())

        print("Average number of distinct features per path")
        print(self.avg_distinct_features())

primary_paths = pd.read_csv('../Outputs/iris_scaled_local_paths_100.csv', header = 0)
secondary_paths = pd.read_csv('../Outputs/iris_scaled_local_paths_per_instance_100.csv', header = 0, index_col=0)
iris = pd.read_csv('../Data/iris_headers.csv', header = 0)
bins = pd.read_csv('../Outputs/iris_scaled_local_bin_labels_100.csv', header = 0)
depths = pd.read_csv('../Outputs/iris_scaled_local_depths_100.csv', header = 0)

# PATHS

primary_paths = primary_paths.values
secondary_paths = secondary_paths.values

bin_vals = bins.values
bin_dict = dict((x[0], float(x[1])) for x in bin_vals)

regex = re.compile('([1-4])([A-Z]+)([01])', re.I)

# DEPTHS

depths = depths.values
depths = depths.flatten()

# DATA AND LABELS

dataset = iris.values
X = dataset[:,0:4].astype(float)
labels = dataset[:,4]

# FEATURE INDEXES

features = [1,2,3,4]

# ARRAYS

ratio_depth_avgpathlen = []
avg_dist_feat = []
average_path_lengths = []

# MAIN

for i in range(150):

    inst = i

    # GET PATHS

    path_list = []

    paths_i = secondary_paths[:,inst]
    paths_i = paths_i[:np.argwhere(pd.isnull(paths_i))[0][0]]
    np.insert(paths_i, 0, primary_paths[inst], axis=0)

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

    metrics = InstanceMetrics(inst, X[inst,:], path_list, weights, labels[inst], features)

    #metrics.display()

    # Calc
    ratio_depth_avgpathlen.append(depths[inst]/metrics.average_path_length())
    avg_dist_feat.append(metrics.avg_distinct_features())
    average_path_lengths.append(metrics.average_path_length())

# print metrics

print("\nMetrics\n")

# ratio depth average path length
print("Ratio of depth to average path length")
print("Mean:",mean(ratio_depth_avgpathlen))
print("Variance:",variance(ratio_depth_avgpathlen))
print("Standard deviation:",stdev(ratio_depth_avgpathlen))
print("Min, max:", min(ratio_depth_avgpathlen), max(ratio_depth_avgpathlen))
p = sns.distplot(ratio_depth_avgpathlen, hist=True, rug=True)
plt.savefig('../Graphs/ratio-depth-avg-path-length.png')
plt.clf()

print("\n")

# average distinct features
print("Average distinct features")
print("Mean:",mean(avg_dist_feat))
print("Variance:",variance(avg_dist_feat))
print("Standard deviation:",stdev(avg_dist_feat))
print("Min, max:",min(avg_dist_feat), max(avg_dist_feat))
p = sns.distplot(avg_dist_feat, hist=True, rug=True)
plt.savefig('../Graphs/avg-distinct-features.png')
plt.clf()

print("\n")

# depths
print("Depths")
print("Mean:",np.mean(depths))
print("Variance:",np.var(depths))
print("Standard deviation:",np.std(depths))
print("Min, max:",min(depths), max(depths))
p = sns.distplot(depths, hist=True, rug=True)
plt.savefig('../Graphs/depths.png')
plt.clf()

print("\n")

# path lengths
print("Average path lengths")
print(average_path_lengths)
