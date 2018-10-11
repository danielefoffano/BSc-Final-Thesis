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
import os

markers = ["s","o","^","p","8","+","h","x"]
colors = ["b","g","r","y","c","m","k","w"]
labels = ["before"]

def crea_cartella(nome_grafo, euristica):
    now = os.getcwd()
    
    os.chdir("..")
    
    if not os.path.isdir(os.getcwd()+"/pdf_results"):
        
        os.mkdir(os.getcwd()+"/pdf_results")
    
    os.chdir("pdf_results")
    
    if not os.path.isdir(os.getcwd()+"/"+nome_grafo):
        
        os.mkdir(os.getcwd()+"/"+nome_grafo)
        
    os.chdir(nome_grafo)
    
    if not os.path.isdir(os.getcwd()+"/"+euristica):
        
        os.mkdir(os.getcwd()+"/"+euristica)
        
    os.chdir(euristica)
    
    if os.path.isfile("risultati.txt"):
        os.remove("risultati.txt")
        
    os.chdir(now)
    

def scrivi_risultato(nome_grafo, euristica, frase):
    
    now = os.getcwd()
    
    os.chdir("..")
    
    os.chdir("pdf_results/" + nome_grafo + "/" + euristica)
    
    f = open("risultati.txt", "a+")
    f.write(frase)
    f.close()
    
    os.chdir(now)

def salva_plot(plot, nome_grafo, nome_file, euristica):
    
    now = os.getcwd()
    
    os.chdir("..")
    
    os.chdir("pdf_results/" + nome_grafo + "/" + euristica)
    
    plot.savefig(nome_file+".pdf")
    
    os.chdir(now)
    
def log_log_graph (XY, markers, colors, labels, graph_name, file_name, euristica):

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
    salva_plot(plt, graph_name, file_name, euristica)
    #plt.show()
    
def distance_graph(distances, graph_name, file_name, euristica):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for key, value in distances.items():
        
        ax1.scatter(key, value, label = "k = {}".format(key))
        
    plt.legend(loc='upper left', ncol = 4)
    plt.ylim(0,1.3)
    plt.xticks(list(distances.keys()), list(distances.keys()))
    salva_plot(plt, graph_name, file_name, euristica)
    #plt.show()

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

#G_init = nx.barabasi_albert_graph(2000, 4) #
graphs = ["facebook_combined", "politician_edges", "tvshow_edges", "athletes"]
euristics = ["degree_based", "indeg_guided"]
name = graphs[0]
euristic = euristics[1]
n = 6

crea_cartella(name, euristic)

f = open("testing_graphs/"+ name +".txt","r")
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

#print("########## Prima dell'anonimizzazione ###########")
scrivi_risultato(name, euristic, "########## Prima dell'anonimizzazione ###########\n\n")

#af.degree_distribution(G_init)

#print("Numero di archi: {}".format(len(nx.edges(G_init))))
scrivi_risultato(name, euristic, "Numero di archi: {}\n\n".format(len(nx.edges(G_init))))

#print("Clustering coefficient: {}".format(nx.average_clustering(G_init)))
scrivi_risultato(name, euristic, "Clustering coefficient: {}\n\n".format(nx.average_clustering(G_init)))

triangle_vertexes = [n for n in nx.triangles(G_init).values()]
triangle_vertexes = reduce(lambda x, y : x + y, triangle_vertexes)
#print("Number of triangles: {}".format(triangle_vertexes/3))
scrivi_risultato(name, euristic, "Number of triangles: {}\n\n".format(triangle_vertexes/3))

originalPrank = nx.pagerank(G_init)
originalPrank = list(originalPrank.values())

#print("Diameter: {}".format(nx.diameter(G_init)))

x1, y1 = calc_xy(G_init)
XY = [(x1,y1)]
every_ser = XY.copy()
every_lab = labels.copy()
distances = defaultdict(int)
direct = nx.is_directed(G_init)
matrix_init = nx.to_numpy_array(G_init)

l = 0
start = time.time()
while(True):
    
    matrix = nx.to_numpy_array(G_init)
    matrix_an, epsilon, sze_idx, irr_pairs, irr_list, classes = af.anonymize(matrix, 2**n, direct, euristic)
    
    try:
        G = nx.from_numpy_matrix(matrix_an)
        
        end = time.time()
        print("\n########## Dopo l'anonimizzazione con k = {} ###########".format(2**n))
        scrivi_risultato(name, euristic, "\n########## Dopo l'anonimizzazione con k = {} ###########\n\n".format(2**n))
        
        #print("\nTempo impiegato : {}".format(end-start))
        scrivi_risultato(name, euristic, "Tempo impiegato : {}\n\n".format(end-start))
        
        #print("\nEpsilon= {}, sze_idx= {}".format(epsilon, sze_idx))
        scrivi_risultato(name, euristic, "Epsilon= {}, sze_idx= {}\n\n".format(epsilon, sze_idx))
        
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
        
        log_log_graph(copy, markers, colors, lab, name, "k_"+str(2**n), euristic) # Salvo il grafico per questo k
        
        
        #print("Numero di archi: {}".format(len(nx.edges(G))))
        scrivi_risultato(name, euristic, "Numero di archi: {}\n\n".format(len(nx.edges(G))))
        #print("Clustering coefficient: {}".format(nx.average_clustering(G)))
        scrivi_risultato(name, euristic, "Clustering coefficient: {}\n\n".format(nx.average_clustering(G)))
        
        triangle_vertexes = [n for n in nx.triangles(G).values()]
        triangle_vertexes = reduce(lambda x, y : x + y, triangle_vertexes)
        #print("Number of triangles: {}".format(triangle_vertexes/3))
        scrivi_risultato(name, euristic, "Number of triangles: {}\n\n".format(triangle_vertexes/3))
        
        #af.degree_distribution(G)

        anonymizedPrank = nx.pagerank(G)
        anonymizedPrank = list(anonymizedPrank.values())
        distance = 1-spatial.distance.cosine(originalPrank, anonymizedPrank)
        distances[2**n] = distance
        #print("Page rank distance: {}".format(distance))
        scrivi_risultato(name, euristic, "Page rank distance: {}\n\n".format(distance))
        
        notR = set()
        
        for (x, y) in irr_list:
            
            notR.add(x)
            notR.add(y)
        
        #print("Numero di partizioni irregolari: {}".format(len(notR)))
        scrivi_risultato(name, euristic, "Numero di partizioni irregolari: {}\n\n".format(len(notR)))
        
        #print("Diameter: {}".format(nx.diameter(G)))
        
        n -= 1
        l = 0
        start = time.time()
        
    except AttributeError:
        print("\n########## Non trovata partizione con k = {} ###########".format(2**n))
        #print("\nTempo impiegato : {}".format(end-start))
        if l < 1500:
            print("Riprovo {}".format(l))
            l += 1
        else:
            l = 0
            n = n-1
            
    if n < 2:
        break
        




#af.degree_distribution(G)
log_log_graph(every_ser, markers, colors, every_lab, name, "every_k", euristic) # Salvo il grafico per tutti i k
distance_graph(distances, name, "Page_rank_distances", euristic) # Salvo il grafico delle distances tra i page rank
