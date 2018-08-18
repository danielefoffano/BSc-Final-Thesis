#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:56:55 2018

@author: daniele
"""
import sys

if './anonymization/' not in sys.path:
    sys.path.insert(1, './anonymization/')

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import anonymization_functions as af
from functools import reduce
#import plotly.plotly as py
#import plotly.graph_objs as go
from plotly.offline import init_notebook_mode#, plot

def log_log_graph (x1, y1, x2, y2):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x1, y1, c='b', marker="s", label='first')
    ax1.scatter(x2, y2, c='r', marker="o", label='second')
    b = plt.gca()
    b.set_xscale("log")
    b.set_yscale("log")
    plt.show()

def calc_xy(G):
    
    dist = defaultdict(int)
    
    for n in G:
        
        dist[G.degree(n)] += 1
        
    x = []
    y = []
    
    for degree, n_nodes in dist.items():
        x.append(degree)
        y.append(n_nodes)
        
    return x, y


init_notebook_mode(connected=True)

G = nx.barabasi_albert_graph(2000, 4) #

if nx.is_directed(G):
    print("Il grafo è diretto.\n")
else:
    print("Il grafo è indiretto.\n")

print("########## Prima dell'anonimizzazione ###########")
af.degree_distribution(G)
#log_log_graph(G)
x1, y1 = calc_xy(G)
print("Numero di archi: {}".format(len(nx.edges(G))))
print("Clustering coefficient: {}".format(nx.average_clustering(G)))
triangle_vertexes = [n for n in nx.triangles(G).values()]
triangle_vertexes = reduce(lambda x, y : x + y, triangle_vertexes)
print("Number of triangles: {}".format(triangle_vertexes/3))
print("Number of maximal cliques: {}".format(len(list(nx.find_cliques(G)))))
#print("Center: {}".format(nx.center(G)))
#print("Diameter: {}".format(nx.diameter(G)))
#print("Radius: {}".format(nx.radius(G)))

#print("Degree Histogram: {}".format(nx.degree_histogram(G)))

#data = [go.Bar(
#            x=[i for i in range(1000)],
#            y=nx.degree_histogram(G)
#    )]
#
#plot(data, filename='basic-bar.html')

matrix = nx.to_numpy_array(G)

#print(matrix)
n = 5
while(True):
    matrix_an, epsilon, sze_idx = af.anonymize(matrix, 2**n, nx.is_directed(G))
    
    try:
        G = nx.from_numpy_matrix(matrix_an)
        print("\nPartizione riuscita con k= {}, epsilon= {}, sze_idx= {}".format(2**n, epsilon, sze_idx))
        break
        
    except AttributeError:
        #print("\nDiminuisco n")
        n -= 1
        


print("\n########## Dopo l'anonimizzazione ###########")

af.degree_distribution(G)
#log_log_graph(G)
x2, y2 = calc_xy(G)
log_log_graph(x1, y1, x2, y2)
print("Numero di archi: {}".format(len(nx.edges(G))))
print("Clustering coefficient: {}".format(nx.average_clustering(G)))
triangle_vertexes = [n for n in nx.triangles(G).values()]
triangle_vertexes = reduce(lambda x, y : x + y, triangle_vertexes)
print("Number of triangles: {}".format(triangle_vertexes/3))
print("Number of maximal cliques: {}".format(len(list(nx.find_cliques(G)))))
#print("Center: {}".format(nx.center(G)))
#print("Diameter: {}".format(nx.diameter(G)))
#print("Radius: {}".format(nx.radius(G)))

#print("Degree Histogram: {}".format(nx.degree_histogram(G)))
#data = [go.Bar(
#            x=[i for i in range(1000)],
#            y=nx.degree_histogram(G)
#    )]
#
#plot(data, filename='basic-bar.html')
