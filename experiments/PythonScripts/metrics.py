import numpy as np
import pandas as pd
import re
from statistics import mean

class Metrics:

    def __init__(self, decision_paths, labels):
        self.decision_paths = decision_paths
        self.labels = labels
        self.overlap_cache = dict()
        self.rule_lengths = []

        for rule in decision_paths:
            self.rule_lengths.append(len(rule))

    def overlap(self, rule1, rule2, data):
        if not repr(rule1) + repr(rule2) in self.overlap_cache:
            self.overlap_cache[repr(rule1) + repr(rule2)] = self.cover_dataset(rule1, data).intersection(self.cover_dataset(rule2, data))
        return len(self.overlap_cache[repr(rule1) + repr(rule2)])

    def cover_point(self, rule, x):
        for predicate in rule:
            if predicate[2]=='0': 
                if x[predicate[0]-1] >= predicate[1]:
                    return False
            else:
                if x[predicate[0]-1] < predicate[1]:
                    return False
        return True

    def cover_dataset(self, rule, data):
        covered_points = set()
        for i, x in enumerate(data):
            if(self.cover_point(rule, x)): covered_points.add(i)
        return covered_points

    def correct_cover(self, rule, label, data):
        full_cover = self.cover_dataset(rule, data)
        correct_cover = set()
        for x in full_cover:
            if labels[x] == label:
                correct_cover.add(x)
        return correct_cover

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

    # 4. Number of classes covered 

    def num_classes_covered(self):
        classes_covered = set()

        for i in range(len(self.labels)):
            classes_covered.add(self.labels[i]) #class of rule

        return len(classes_covered)

    ## B. WITH INPUT DATA ##

    # 5. Intra-class overlap

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

    # 6. Inter-class overlap

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
            #print(cover,full_cover)
            if full_cover==0:
                covers.append(0)
            else:
                covers.append(cover/full_cover)
        return covers


paths = pd.read_csv('../Outputs/iris_scaled_local_paths_100.csv', header = 0)
iris = pd.read_csv('../Data/iris_headers.csv', header = 0)
bins = pd.read_csv('../Outputs/iris_scaled_local_bin_labels_100.csv', header = 0)
depths = pd.read_csv('../Outputs/iris_scaled_local_depths_100.csv', header = 0)

# PATHS

paths = paths.values
bin_vals = bins.values

bin_dict = dict((x[0], float(x[1])) for x in bin_vals)

regex = re.compile('([1-4])([A-Z]+)([01])', re.I)
    
path_list = []

for path in paths:
    nodes = path[0].split(",")
    newpath = []
    for node in nodes:
        matchobj =  re.match(regex, node)
        newpath.append((int(matchobj.group(1)), bin_dict[matchobj.group(2)], matchobj.group(3)))
    path_list.append(newpath)

# DEPTHS

depths = depths.values
depths = depths.flatten()

# DATA AND LABELS

dataset = iris.values
X = dataset[:,0:4].astype(float)
labels = dataset[:,4]

# MAIN

metrics = Metrics(path_list, labels)

print("Decision set size")
print(metrics.decision_paths_size())
print("Decision set length")
print(metrics.decision_paths_length())
print("Average rule length")
print(metrics.average_rule_length())
print("Inter class overlap")
print(metrics.interclass_overlap(X))
print("Intra class overlap")
print(metrics.intraclass_overlap(X))
print("Total number of classes covered")
print(metrics.num_classes_covered())
print("Correct cover")
print(metrics.total_correct_cover(X))
print("Incorrect cover")
print(metrics.total_incorrect_cover(X))
print("Depths")
print(depths)
print("Correct cover")
print(metrics.total_correct_cover_rule(X))