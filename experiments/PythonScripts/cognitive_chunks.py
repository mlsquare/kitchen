import pandas as pd
import math

def get_cognitive_chunks(path):
    chunks = [[float('-inf'), float('inf')] for _ in range(4)]
    for node in path:
        if node[2]=='0' and chunks[node[0]-1][1]>node[1]:
            chunks[node[0]-1][1] = node[1]
        if node[2]=='1' and chunks[node[0]-1][0]<node[1]:
            chunks[node[0]-1][0] = node[1]
    
    index = [i+1 for i,x in enumerate(chunks) if x!=[float('-inf'), float('inf')]]
    for i,x in enumerate(chunks):
        x.insert(0, i+1)
    chunks = [x for i,x in enumerate(chunks) if i+1 in index]
    return chunks

def get_cognitive_chunks_round(path):
    chunks = [[float('-inf'), float('inf')] for _ in range(4)]
    for node in path:
        print(node[1])
        if node[2]=='0' and chunks[node[0]-1][1]>math.floor(node[1]*10000)/10000:
            chunks[node[0]-1][1] = math.floor(node[1]*10000)/10000
        if node[2]=='1' and chunks[node[0]-1][0]<math.floor(node[1]*10000)/10000:
            chunks[node[0]-1][0] = math.floor(node[1]*10000)/10000
    
    index = [i+1 for i,x in enumerate(chunks) if x!=[float('-inf'), float('inf')]]
    for i,x in enumerate(chunks):
        x.insert(0, i+1)
    chunks = [x for i,x in enumerate(chunks) if i+1 in index]
    return chunks
