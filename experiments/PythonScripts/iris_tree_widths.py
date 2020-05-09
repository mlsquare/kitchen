import pandas as pd
import numpy as np

filename = '../R/local_dt_info_iris_scaled/frame_'
ext = '.csv'

max_widths = []

for i in range(1,151): # RANGE
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

pd.DataFrame(max_widths).to_csv("../Outputs/iris_tree_widths.csv", header = ['width','at-depth'], index = False)
