#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:15:36 2019

@author: daniele
"""

import collections
import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx
import os
import re
import scipy as sp
from collections import defaultdict
from scipy.stats import spearmanr
from scipy import spatial

def leggi_grafi_anonimizzati(nome_grafo, euristica):
    
    now = os.getcwd()
    
    os.chdir("archi_grafi" + "/" + nome_grafo)
    
    '''Importo il grafo originale'''
    
    original = nx.Graph()
    
    f = open(nome_grafo + ".txt","r")
    
    couples = []
    edges = []

    couples = [line.split() for line in f.readlines()]
    
    f.close()
    
    for couple in couples:
        
        edges.append((int(couple[0]), int(couple[1])))
    
    original.add_edges_from(edges)
    
    '''Importo i grafi anonimizzati con l'euristica desiderata'''
    
    os.chdir(euristica)
    
    anonim = defaultdict(list)
    nodes = original.nodes()
    
    for file in os.listdir():
        
        if(file.endswith('.txt')):
            
            k = re.findall(r"_(\d+)_", file)[0]
            
            #print(file)
            #print(k)
            
            f = open(file,"r")
                
            G=nx.Graph()
            G.add_nodes_from(nodes)
            
            couples = []
            edges = []
    
            couples = [line.split() for line in f.readlines()]
            
            f.close()
            
            for couple in couples:
                
                edges.append((int(couple[0]), int(couple[1])))
            
            G.add_edges_from(edges)
            
            anonim[int(k)].append(G.copy())
            
            
            
    os.chdir(now)
    
    return original, anonim

def salva_plot(plot, nome_file, nome_grafo):
    
    now = os.getcwd()
    
    os.chdir("archi_grafi/" + nome_grafo)
    
    plot.savefig(nome_file+".png",format="PNG", bbox_inches="tight")
    
    os.chdir(now)

def distance_graph(spmc, cosine, graph_name, euristica):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    std = [(np.std(spmc[key]), len(spmc[key])) for key in sorted(spmc.keys())]
    mean = [np.mean(spmc[key]) for key in sorted(spmc.keys())] #y
    std_err = [dev/math.sqrt(n) for dev, n in std] #e
    lbl = [key for key in sorted(spmc.keys())]
    
    plt.xticks(range(len(mean)), lbl)
    plt.ylim(0.8,1.0)
    plt.xlabel("l")
    plt.ylabel("Avg Similarity Score")
    
    ax1.errorbar(range(len(mean)), mean, std_err, marker='o', markersize = 3, linewidth = 1, capsize = 6, label = "Spearman")
    
    std = [(np.std(cosine[key]), len(cosine[key])) for key in sorted(cosine.keys())]
    mean = [np.mean(cosine[key]) for key in sorted(cosine.keys())] #y
    std_err = [dev/math.sqrt(n) for dev, n in std] #e
    lbl = [key for key in sorted(cosine.keys())]
    
    ax1.errorbar(range(len(mean)), mean, std_err, marker='o', markersize = 3, linewidth = 1, capsize = 6, color = 'orange', label = "Cosine")
    
    plt.legend(loc='upper left')
    #plt.show()
    salva_plot(plt, graph_name + "_" + euristica + "_PageRank_Similarity", graph_name)
    
def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return sp.stats.entropy(p,m, base=base)/2. +  sp.stats.entropy(q, m, base=base)/2.


nome_grafo = 'politician_edges'
euristica = 'indeg_guided'

original_graph, anonim_graphs = leggi_grafi_anonimizzati(nome_grafo, euristica)

all_jsd = defaultdict(list)
jsd_mean = defaultdict(int)
jsd_stde = defaultdict(int)

degree_sequence = sorted([d for n, d in original_graph.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)

for k in anonim_graphs.keys():
    
    for graph in anonim_graphs[k]:
    
        an_degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
        an_degreeCount = collections.Counter(an_degree_sequence)
        
        for deg in sorted(degreeCount.keys()):
    
            if not(deg in an_degreeCount):
                
                an_degreeCount[deg] = 0
                
        for deg in sorted(an_degreeCount.keys()):
            
            if not(deg in degreeCount):
                
                degreeCount[deg] = 0
                
        original_nodes = [degreeCount[deg] for deg in sorted(degreeCount.keys())]
        an_nodes = [an_degreeCount[deg] for deg in sorted(an_degreeCount.keys())]
        
        
        all_jsd[k].append(jsd(original_nodes, an_nodes))
    
    jsd_mean[k] = np.mean(all_jsd[k])
    jsd_stde[k] = np.std(all_jsd[k])/math.sqrt(len(anonim_graphs[k]))
    
    print('Per k = {} la JSD media è {} con uno stdErr di {}'.format(k, jsd_mean[k], jsd_stde[k]))




#print(original_graph.edges())

#originalPrank = nx.pagerank(original_graph)
#originalPrank = list(originalPrank.values())
#
#spmc_correlations = defaultdict(list)
#cosines = defaultdict(list)
#
#for k in anonim_graphs.keys():
#    
#    #avg_spmc = 0
#    #avg_distance = 0
#    
#    for graph in anonim_graphs[k]:
#        
#        anonimPrank = nx.pagerank(graph)
#        anonimPrank = list(anonimPrank.values())
#        
#        #avg_spmc += spearmanr(originalPrank, anonimPrank)[0]
#        spmc_correlations[k].append(spearmanr(originalPrank, anonimPrank)[0])
#        cosines[k].append(1-spatial.distance.cosine(originalPrank, anonimPrank))
#        
#    #spmc_correlations[k] = avg_spmc/len(anonim_graphs[k])
#    #print(spmc_correlations)
#    #cosines[k] = avg_distance/len(anonim_graphs[k])
#    
#
#distance_graph(spmc_correlations, cosines, nome_grafo, euristica)

#plt.clf()



#degree_sequence = sorted([d for n, d in original_graph.degree()], reverse=True)
#plt.hist(degree_sequence, bins=25, cumulative=True, density=True, histtype='step', label = 'Original graph')
#
#i = 9
#for k in sorted(anonim_graphs.keys(), reverse = True):
#    
#    i = i-2
#    deg_seq_an = sorted([d for n, d in anonim_graphs[k][0].degree()], reverse=True)
#    plt.hist(deg_seq_an, bins=25, cumulative=True, density=True, histtype='step', label = 'l = '+str(k))
#
#plt.legend(loc = 'lower right')
#plt.ylim(bottom = 0.4)
#plt.ylabel('CDF')
#plt.xlabel('Degree')
#salva_plot(plt, nome_grafo + '_' + euristica + '_' + 'distribution_distance', nome_grafo)
#plt.show()
        
    