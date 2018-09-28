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
from scipy import spatial
import time
import operator
#import plotly.plotly as py
#import plotly.graph_objs as go
from plotly.offline import init_notebook_mode#, plot

markers = ["s","o","^","p","8","+","h","x"]
colors = ["b","g","r","y","c","m","k","w"]
labels = ["before"]

def log_log_graph (XY, markers, colors, labels):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    i=0
    for (x, y) in XY:
        ax1.scatter(x, y, c=colors[i], marker=markers[i], label=labels[i])
        i = i + 1
    b = plt.gca()
    b.set_xscale("log")
    b.set_yscale("log")
    plt.legend(loc='upper right')
    plt.show()
    
def distance_graph(distances):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for key, value in distances.items():
        
        ax1.scatter(key, value, label = "k = {}".format(key))
        
    plt.legend(loc='upper left', ncol = 4)
    plt.ylim(0,1.3)
    plt.xticks(list(distances.keys()), list(distances.keys()))
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

def reverse_dictionary(d):
    
    n = defaultdict(int)
    
    for key, value in d.items():
        
        n[value] += 1
        
    return n

def check_high_degree(nodes, classes, irr_list):
    
    in_irr = 0
    tot = len(nodes)
    
    for pair in irr_list:
        
        for node in nodes:
            
            if classes[node] == pair[0] or classes[node] == pair[1]:
                
                in_irr += 1
                nodes.remove(node)
            
    print("\n{} hub su {} sono in partizioni irregolari.".format(in_irr, tot))


init_notebook_mode(connected=True)

#G = nx.barabasi_albert_graph(2000, 4) #
f = open("testing_graphs/athletes.txt","r")
#f = open("testing_graphs/tvshow_edges.txt","r")
#f = open("testing_graphs/facebook_combined.txt","r")
#f = open("testing_graphs/politician_edges.txt","r")
G_init=nx.Graph()

nodes = set()
couples = []
edges = []

couples = [line.split() for line in f.readlines()]
for couple in couples:
    
    edges.append((int(couple[0]), int(couple[1])))
    edges.append((int(couple[1]), int(couple[0])))

for (x,y) in edges:
    
    nodes.add(x)
    nodes.add(y)

G_init.add_nodes_from(list(nodes))
G_init.add_edges_from(edges)

#print(nx.adjacency_matrix(G))

print("numero di nodi: {}".format(len(G_init.nodes)))

if nx.is_directed(G_init):
    print("Il grafo è diretto.\n")
else:
    print("Il grafo è indiretto.\n")

print("########## Prima dell'anonimizzazione ###########")
af.degree_distribution(G_init)
#log_log_graph(G)
x1, y1 = calc_xy(G_init)

print("Numero di archi: {}".format(len(nx.edges(G_init))))
print("Clustering coefficient: {}".format(nx.average_clustering(G_init)))

triangle_vertexes = [n for n in nx.triangles(G_init).values()]
triangle_vertexes = reduce(lambda x, y : x + y, triangle_vertexes)
print("Number of triangles: {}".format(triangle_vertexes/3))

#print("Number of maximal cliques: {}".format(len(list(nx.find_cliques(G)))))

originalPrank = nx.pagerank(G_init)
originalPrank = list(originalPrank.values())

#print("Page rank: {}".format(originalPrank.values()))

#print("Center: {}".format(nx.center(G)))
#print("Diameter: {}".format(nx.diameter(G)))
#print("Radius: {}".format(nx.radius(G)))

matrix_init = nx.to_numpy_array(G_init)

#print(matrix)
n = 4

XY = [(x1,y1)]
every_ser = XY.copy()
every_lab = labels.copy()
distances = defaultdict(int)
direct = nx.is_directed(G_init)

l = 0
while(True):
    start = time.time()
    matrix = nx.to_numpy_array(G_init)
    matrix_an, epsilon, sze_idx, irr_pairs, irr_list, classes = af.anonymize(matrix, 2**n, direct)
    end = time.time()
    try:
        G = nx.from_numpy_matrix(matrix_an)
        
        print("\n########## Dopo l'anonimizzazione con k = {} ###########".format(2**n))
        print("\nTempo impiegato : {}".format(end-start))
        print("\nEpsilon= {}, sze_idx= {}".format(epsilon, sze_idx))
        
        nodes = G.degree()
        nodes = list(nodes)
        nodes.sort(key = operator.itemgetter(1), reverse = True)
        nodes = [x for (x,y) in nodes]
        check_high_degree(nodes[:100], classes, irr_list)
        
        
        copy = XY.copy()
        copy.append((calc_xy(G)))
        every_ser.append((calc_xy(G)))
        lab = labels.copy()
        lab.append("k = "+str(2**n))
        every_lab.append("k = "+str(2**n))
        log_log_graph(copy, markers, colors, lab)
        
        
        print("Numero di archi: {}".format(len(nx.edges(G))))
        print("Clustering coefficient: {}".format(nx.average_clustering(G)))
        
        triangle_vertexes = [n for n in nx.triangles(G).values()]
        triangle_vertexes = reduce(lambda x, y : x + y, triangle_vertexes)
        print("Number of triangles: {}".format(triangle_vertexes/3))
        
        #print("Number of maximal cliques: {}".format(len(list(nx.find_cliques(G)))))
        
        af.degree_distribution(G)

        anonymizedPrank = nx.pagerank(G)
        anonymizedPrank = list(anonymizedPrank.values())
        distance = 1-spatial.distance.cosine(originalPrank, anonymizedPrank)
        distances[2**n] = distance
        print("Page rank distance: {}".format(distance))
        
        notR = set()
        
        for (x, y) in irr_list:
            
            notR.add(x)
            notR.add(y)
        
        print("Numero di partizioni irregolari: {}".format(len(notR)))
        
        #print("Center: {}".format(nx.center(G)))
        #print("Diameter: {}".format(nx.diameter(G)))
        #print("Radius: {}".format(nx.radius(G)))
#        if l < 1000:
#            print("Riprovo {}".format(l))
#            l += 1
#        else:
#            l = 0
#            n = n-1
            
        n -= 1
        l = 0
        
    except AttributeError:
        print("\n########## Non trovata partizione con k = {} ###########".format(2**n))
        print("\nTempo impiegato : {}".format(end-start))
        if l < 1500:
            print("Riprovo {}".format(l))
            l += 1
        else:
            l = 0
            n = n-1
    
    #n = n-1
    if n < 2:
        break
        




#af.degree_distribution(G)
log_log_graph(every_ser, markers, colors, every_lab)
distance_graph(distances)
