import numpy as np
import pandas as pd
import re
from statistics import mean
import json
import argparse
import pickle
from PythonScripts.cognitive_chunks import get_cognitive_chunks_round, get_cognitive_chunks_round

class Metrics:

    def __init__(self, decision_paths, labels, features, factors=None):
        self.decision_paths = decision_paths
        self.labels = labels
        self.features = features
        self.overlap_cache = dict()
        self.rule_lengths = []
        if factors is None:
            self.factors = []
            self.factor_set = dict()
        else:
            self.factors = factors
            self.factor_set = dict([(int(f),i) for i,f in enumerate(factors[0,:])])

        for rule in decision_paths:
            self.rule_lengths.append(len(rule))

    def overlap(self, rule1, rule2, data):
        if not repr(rule1) + repr(rule2) in self.overlap_cache:
            self.overlap_cache[repr(rule1) + repr(rule2)] = self.cover_dataset(rule1, data).intersection(self.cover_dataset(rule2, data))
        return len(self.overlap_cache[repr(rule1) + repr(rule2)])

    def cover_point(self, rule, x):
        for predicate in rule:
            if predicate[0] in self.factor_set:
                if not x[predicate[0]-1] == predicate[1]:
                    return False
            elif predicate[2]=='0': 
                if x[predicate[0]-1] >= float(predicate[1]):
                    return False
            else:
                if x[predicate[0]-1] < float(predicate[1]):
                    return False
        return True

    def cover_dataset(self, rule, data):
        covered_points = set()
        for i, x in enumerate(data):
            if self.cover_point(rule, x): covered_points.add(i)
        return covered_points

    def correct_cover(self, rule, label, data):
        pass
        # full_cover = self.cover_dataset(rule, data)
        # correct_cover = set()
        # for x in full_cover:
        #     if self.labels[x] == label:
        #         correct_cover.add(x)
        # return correct_cover

    def incorrect_cover(self, rule, label, data):
        full_cover = self.cover_dataset(rule, data)
        correct_cover = self.correct_cover(rule, label, data)
        return full_cover.difference(correct_cover)

    
    #####   INTERPRETABILITY METRICS    #####


    ### RULE METRICS ###

    # 1. Length of the rule

    def rule_length(self, rule):
        return len(rule)

    # 2. Number of distinct features of a rule

    def rule_distinct_features(self, rule):
        features = [predicate[0] for predicate in rule]
        distinct_features = set(features)
        return distinct_features


    ### DECISION SET METRICS ###


    ## A. WITHOUT INPUT DATA ##

    # 1. Size of decision set

    def decision_paths_size(self):
        return len(self.decision_paths)

    # 2. Sum of length of rules of decision set

    def decision_paths_length(self):
        return sum(self.rule_lengths)

    # 3. Average rule length

    def average_rule_length(self):
        return mean(self.rule_lengths)

    # 4. Number of classes/bins covered 

    def num_classes_covered(self):
        pass

    # 5. Average frequency of each feature in all paths at all depths

    def mean_rank(self):
        freq = np.zeros(len(self.features))
        for path in self.decision_paths:
            for pred in path:
                freq[pred[0]-1] += 1
        freq /= freq.sum(axis=0, keepdims=True)
        #freq = np.true_divide(freq, freq.sum(axis=0, keepdims=True))
        return freq

    # 6. (Rooted at k) Frequency of all features occurring at a depth in all paths 
    # Returns an array of frequencies of features at depth d

    def frequency_at_depth(self, depth):
        freq = np.zeros(len(self.features))
        for path in self.decision_paths:
            if len(path)>depth: # depth is valid
                freq[path[depth][0]-1] += 1
        s = freq.sum(axis=0, keepdims=True)
        if s!=0:
            freq /= s
        else:
            freq = np.zeros(len(self.features))
        return freq

    # 7. Frequency of all features occuring at all depths
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

    #8. Average number of distinct features

    def average_distinct_features(self):
        num_distinct_features = []
        for path in self.decision_paths:
            num_distinct_features.append(len(self.rule_distinct_features(path)))
        return mean(num_distinct_features)

    ## B. WITH INPUT DATA ##

    # 1. Intra-class overlap

    def intraclass_overlap(self, data):
        overlap_intraclass_sum = 0

        for i, r1 in enumerate(self.decision_paths):
            for j, r2 in enumerate(self.decision_paths):
                if i >= j:
                    continue

                if self.labels[i] == self.labels[j]: #if class labels are equal
                    overlap_tmp = self.overlap(r1, r2, data)
                    overlap_intraclass_sum += overlap_tmp     

        return overlap_intraclass_sum

    # 2. Inter-class overlap

    def interclass_overlap(self, data):
        overlap_interclass_sum = 0

        for i, r1 in enumerate(self.decision_paths):
            for j, r2 in enumerate(self.decision_paths):
                if i >= j:
                    continue

                if self.labels[i] != self.labels[j]: #if class labels are equal
                    overlap_tmp = self.overlap(r1, r2, data)
                    overlap_interclass_sum += overlap_tmp     

        return overlap_interclass_sum



    
    #####   ACCURACY METRICS    #####


    ### RULE METRICS ###

    # 1. Incorrect cover

    def rule_incorrect_cover(self, rule, label, data):
        incorrect_cover_points = self.incorrect_cover(rule, label, data)
        full_cover_points = self.cover_dataset(rule, data)
        return len(incorrect_cover_points), len(full_cover_points)

    # 2. Correct cover

    def rule_correct_cover(self, rule, label, data):
        correct_cover_points = self.correct_cover(rule, label, data)
        full_cover_points = self.cover_dataset(rule, data)
        return len(correct_cover_points), len(full_cover_points)


    ### DECISION SET METRICS ###

    # 1. Fraction incorrect cover

    def total_incorrect_cover(self, data):
        total_incorrect, total_full_cover = 0, 0
        for i, rule in enumerate(self.decision_paths):
            cover, full_cover = self.rule_incorrect_cover(rule, self.labels[i], data)
            total_incorrect += cover
            total_full_cover += full_cover
        return total_incorrect/total_full_cover

    # 2. Fraction correct cover

    def total_correct_cover(self, data):
        total_correct, total_full_cover = 0, 0
        for i, rule in enumerate(self.decision_paths):
            cover, full_cover = self.rule_correct_cover(rule, self.labels[i], data)
            total_correct += cover
            total_full_cover += full_cover
        return total_correct/total_full_cover

    #3. Return incorrect cover per rule

    def total_incorrect_cover_rule(self, data):
        covers = []
        for i, rule in enumerate(self.decision_paths):
            cover, full_cover = self.rule_incorrect_cover(rule, self.labels[i], data)
            if full_cover==0:
                covers.append("NA")
            else:
                covers.append(cover/full_cover)
        return covers

    #4. Return correct cover per rule

    def total_correct_cover_rule(self, data):
        covers = []
        for i, rule in enumerate(self.decision_paths):
            cover, full_cover = self.rule_correct_cover(rule, self.labels[i], data)
            if full_cover==0:
                covers.append(0)
            else:
                covers.append(cover/full_cover)
        return covers

    def display(self, X):
        print("METRICS")
        print("Decision set size")
        print(self.decision_paths_size())
        print("Decision set length")
        print(self.decision_paths_length())
        print("Average rule length")
        print(self.average_rule_length())
        print("Average distinct features")
        print(self.average_distinct_features())
        # print("Inter class overlap")
        # print(self.interclass_overlap(X))
        # print("Intra class overlap")
        # print(self.intraclass_overlap(X))
        print("Total number of classes covered")
        print(self.num_classes_covered())
        print("Correct cover")
        print(self.total_correct_cover(X))
        print("Incorrect cover")
        print(self.total_incorrect_cover(X))
        print("Mean rank")
        print(self.mean_rank())
        print("RAK")
        print(self.frequency_at_all_depths())


class ClassificationMetrics(Metrics):
    def __init__(self, decision_paths, labels, features, factors=None):
        super().__init__(decision_paths, labels, features, factors)
    
    def correct_cover(self, rule, label, data):
        full_cover = self.cover_dataset(rule, data)
        correct_cover = set()
        for x in full_cover:
            if self.labels[x] == label:
                correct_cover.add(x)
        return correct_cover

    def num_classes_covered(self):
        classes_covered = set()

        for i in range(len(self.labels)):
            classes_covered.add(self.labels[i]) #class of rule

        return len(classes_covered)
        

class RegressionMetrics(Metrics):

    def __init__(self, decision_paths, labels, features, factors=None, bins=None):
        super().__init__(decision_paths, labels, features, factors)
        self.bins = bins
        self.binned_labels = np.digitize(self.labels, self.bins)
    
    def correct_cover(self, rule, label, data):
        binned_label = np.digitize([label], self.bins)
        full_cover = self.cover_dataset(rule, data)
        correct_cover = set()
        for x in full_cover:
            if self.binned_labels[x] == binned_label:
                correct_cover.add(x)
        return correct_cover

    def num_classes_covered(self):
        classes_covered = set()

        for i in range(len(self.binned_labels)):
            classes_covered.add(self.binned_labels[i]) #binned label of rule

        return len(classes_covered)


class Comparison:

    def __init__(self, original, perturbed):
        self.original = original
        self.perturbed = perturbed
        self.num_features = len(self.original.features)

    def change_of_class(self):
        op = []
        for i in range(len(self.original.decision_paths)):
            new_op = []
            c1 = self.original.labels[i]
            c2 = self.perturbed.labels[i]
            p1 = self.original.decision_paths[i]
            p2 = self.perturbed.decision_paths[i]

            if c1!=c2:
                new_op.append(c1)
                new_op.append(c2)
                new_op.append(get_cognitive_chunks_round(p1, self.num_features))
                if p1==p2:
                    new_op.append('No change')
                else:
                    new_op.append(get_cognitive_chunks_round(p2, self.num_features))
            op.append(new_op)
        op = pd.DataFrame(op, columns=['Original label','New label', 'Original nodes','Changed nodes'])
        return op

    def print_change_of_class(self):
        changes = self.change_of_class()
        print(changes)

def main(dataset):

    # DATASET

    with open('../Configs/'+dataset+'.json') as config_file:
        config = json.load(config_file)

    paths = pd.read_csv('../Outputs/'+config['perturbed_paths'], header = 0)
    dataset = pd.read_csv('../Data/'+config['filtered_data_with_headers'], header = 0)
    perturbed_dataset = pd.read_csv('../Data/'+config['perturbed_data'], header = 0)
    bins = pd.read_csv('../Outputs/'+config['perturbed_local_bins'], header = 0)
    depths = pd.read_csv('../Outputs/'+config['tree_depths'], header = 0)
    
    # PATHS

    paths = paths.values

    bin_vals = bins.values

    if 'factors' in config:
        bin_dict = dict((x[0], x[1]) for x in bin_vals) #.replace(')', '').replace('(', '')
    else:
        bin_dict = dict((x[0], float(x[1])) for x in bin_vals)

    regex = re.compile(config['path_regex'], re.I)

    path_list = []

    for i in range(2):
        temp = []
        for path in paths[:,i]:
            nodes = path.split(",")
            newpath = []
            for node in nodes:
                matchobj =  re.match(regex, node)
                newpath.append((int(matchobj.group(1)), bin_dict[matchobj.group(2)], matchobj.group(3)))
            temp.append(newpath)
        path_list.append(temp)

    # DEPTHS

    depths = depths.values
    depths = depths.flatten()

    # FEATURE INDEXES

    features = np.arange(1,config['num_features']+1)

    # DATA AND LABELS

    dataset = dataset.values[0:config['sample']]
    X = dataset[:,0:config['num_features']]
    labels = dataset[:,config['target_col']-1]

    perturbed_dataset = perturbed_dataset.values[0:config['sample']]
    perturbed_X = perturbed_dataset[:,0:config['num_features']]
    perturbed_labels = perturbed_dataset[:,config['target_col']]

    # FACTORS

    factors = None
    if 'factors' in config:
        factors = pd.read_csv('../Outputs/'+config['factors'], header = 0)
        factors = factors.values    
        

    # BINS
    ## Labels are binned

    if config['type'] == 'regression':
        with open ('../Outputs/'+config['label_bins'], 'rb') as fp:
            label_bins = pickle.load(fp)

    # METRICS

    if config['type'] == 'classification':
        metrics = ClassificationMetrics(path_list[0], labels, features, factors)
        perturbed_metrics = ClassificationMetrics(path_list[1], perturbed_labels, features, factors)
    elif config['type'] == 'regression':
        metrics = RegressionMetrics(path_list[0], labels, features, factors, label_bins)
        perturbed_metrics = RegressionMetrics(path_list[1], perturbed_labels, features, factors, label_bins)
    else:
        print(("Type {} not supported").format(config['type']))
        exit(0)

    # DISPLAY

    metrics.display(X)
    perturbed_metrics.display(perturbed_X)

    # COMPARISON

    if 'factors' not in config:
        comp = Comparison(metrics, perturbed_metrics)
        comp.print_change_of_class()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Instance metrics', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, required=True, help = 'name of the dataset')
    (args, _) = parser.parse_known_args()

    main(args.dataset)
