import pandas as pd
import re
import pickle

paths = pd.read_csv('../Outputs/iris_scaled_local_paths_100.csv', header = 0)
paths = paths.values
bins = pd.read_csv('../Outputs/iris_scaled_local_bin_labels_100.csv', header = 0)
bin_vals = bins.values

bin_dict = dict((x[0], x[1]) for x in bin_vals)

regex = re.compile('([1-4])([A-Z]+)([01])', re.I)
    
path_list = []

for path in paths:
    nodes = path[0].split(",")
    newpath = []
    for node in nodes:
        matchobj =  re.match(regex, node)
        newpath.append((matchobj.group(1), bin_dict[matchobj.group(2)], matchobj.group(3)))
    path_list.append(newpath)

pickle.dump(path_list, open('../Outputs/path_list.dump', 'wb'))
#pl = pickle.load(open('path_list.dump', 'rb'))
