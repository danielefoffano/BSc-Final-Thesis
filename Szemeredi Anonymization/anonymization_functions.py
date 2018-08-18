# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import random as rdm
from collections import defaultdict
from power_partition import power_partition
import networkx as nx


'''
    Function to make an oriented graph, a non oriented graph.
    Not used, so far.
'''

def make_not_oriented(G):
    
    for x, y in nx.edges(G):
        
        if (y, x) not in nx.edges(G):
            
            G.add_edge(y,x)
            
    return G

'''
    Function to calculate a general nodes grade distribution
    (Useful just to see if the graph has hubs, as it should)
'''

def degree_distribution(G):
    
    less_ten = 0
    
    ten_twenty = 0
    
    twenty_thirty = 0
    
    thirty_fourty = 0
    
    more_fourty = 0
    
    for node, degree in nx.degree(G, range(1000)):
        
        if(degree<10):
        
            less_ten += 1
            
        if(degree >= 10 and degree < 20):
            
            ten_twenty += 1
            
        if(degree >= 20 and degree < 30):
            
            twenty_thirty += 1
        
        if(degree >= 30 and degree < 40):
            
            thirty_fourty += 1
            
        if(degree >= 40):
            
            more_fourty += 1
        
    print("\nNodi con degree minore di 10: {}".format(less_ten))
    print("Nodi con degree tra 10 e 20: {}".format(ten_twenty))
    print("Nodi con degree tra 20 e 30: {}".format(twenty_thirty))
    print("Nodi con degree tra 30 e 40: {}".format(thirty_fourty))
    print("Nodi con degree maggiore di 40: {}".format(more_fourty))


'''
    Function to calculate the inner edges density of a set of nodes
    Function parameters:
        * subset : the set of nodes
        * adj_matrix : the adjacency matrix of the graph
        * oriented : boolean value. True if the graph is oriented, false otherwise.

    Function output:
        * inside_edges/possible_edges : also known as the set of nodes density
        * edges : a list of all the set of nodes possible inner edges
'''
def density (subset, adj_matrix, oriented):
    
    n = len(subset)
    inside_edges = 0
    possible_edges = (n*(n-1)) #non è diviso 2 perchè considero tutte le coppie di vertici, e non il numero di archi
    edges = []
    
    
    if not oriented:
        while(len(subset) != 0):
            node1 = subset.pop()
            
            for node2 in subset:
                edges.append((node1, node2)) #considero soltanto un verso per randomizzare meglio entrambi con Erdos-Renyi
                if(adj_matrix[node1][node2] == 1):
                    inside_edges += 2 #aggiungo due perchè considero entrambi i versi dell'arco
    else:
        while(len(subset) != 0):
            node1 = subset.pop()
            
            for node2 in subset:
                
                edges.append((node1, node2))
                edges.append((node2, node1))
                
                if(adj_matrix[node1][node2] == 1):
                    inside_edges += 1
                    
                if(adj_matrix[node2][node1] == 1):
                    inside_edges += 1
        
    return inside_edges/possible_edges, edges

'''
    Function using the Erdos-Renyi model to randomize a set of edges of a group of nodes.
    The probability used is the inner edges density before the randomization.

    Function parameters:
        * p : the probability used by the Erdos-Renyi model
        * edges : the set of edges to randomize
        * adj_matrix : the adjacency matrix of the graph
        * oriented : boolean value. True if the graph is oriented, false otherwise.

    Function output:
        * adj_matrix : the adjacency matrix of the graph, with the randomized edges
'''

def Erdos_Renyi(p, edges, adj_matrix, oriented):
    
    if not oriented:
        
        for (n1, n2) in edges:
            
            i = rdm.random()
            
            if(i <= p):
                adj_matrix[n1][n2] = 1
                adj_matrix[n2][n1] = 1
            else:
                adj_matrix[n1][n2] = 0
                adj_matrix[n2][n1] = 0
    else:
        
        for (n1, n2) in edges:
            
            i = rdm.random()
            
            if(i <= p):
                adj_matrix[n1][n2] = 1
            else:
                adj_matrix[n1][n2] = 0
        

    return adj_matrix

'''
    Function using the Erdos-Renyi model to randomize the edges going from one group of nodes to another
    (i.e.: that have one vertex in the first group and the other in the second one).
    The probability used by the model is the density of the edges between these two groups.

    Function parameters:
        * group1 : the first group of nodes
        * group2 : the second group of nodes
        * adj_matrix : the adjacency matrix of the graph
        * oriented : boolean value. True if the graph is oriented, false otherwise.

    Function output:
        * adj_matrix : the adjacency matrix of the graph, with the randomized edges
'''


def anonymize_irr_couple(group1, group2, adj_matrix, oriented):
    
    possible_edges = len(group1) * len(group2) * 2 #è il doppio perchè considero tutte le coppie di vertici, e non il numero di archi
    between_edges = 0
    
    if not oriented:
        
        for n1 in group1:
            
            for n2 in group2:
                
                if(adj_matrix[n1][n2] == 1):
                    
                    between_edges += 2 #aggiungo due perchè considero entrambi i versi dell'arco
                    
        density = between_edges/possible_edges
        
        for n1 in group1:
            
            for n2 in group2:
                
                i = rdm.random()
            
                if(i <= density):
                    adj_matrix[n1][n2] = 1
                    adj_matrix[n2][n1] = 1
                else:
                    adj_matrix[n1][n2] = 0
                    adj_matrix[n2][n1] = 0
                    
    else:
        
        for n1 in group1:
            
            for n2 in group2:
                
                if(adj_matrix[n1][n2] == 1):
                    
                    between_edges += 1 
                    
                if(adj_matrix[n2][n1] == 1):
                    
                    between_edges += 1
                    
        density = between_edges/possible_edges
        
        for n1 in group1:
            
            for n2 in group2:
                
                i = rdm.random()
            
                if(i <= density):
                    adj_matrix[n1][n2] = 1
                else:
                    adj_matrix[n1][n2] = 0
                    
                i = rdm.random() #eseguo due volte la randomizzazione perchè randomizzo sia l'arco in un verso che nel verso opposto
            
                if(i <= density):
                    adj_matrix[n2][n1] = 1
                else:
                    adj_matrix[n2][n1] = 0
        
                    
    return adj_matrix

'''
    Function to anonymize a graph using the Szemeredi regularity lemma and the Erdos-Renyi model.

    Function parameters:

        * adj_matrix: the adjacency matrix of the graph
        * k : the number of partitions desired by the user (only powers of two are accepted)
        * oriented : boolean value. True if the graph is oriented, false otherwise.

    Function output:

        * adj_matrix : the adjacency matrix of the randomized graph
'''


def anonymize(adj_matrix, k, oriented):   
    
    '''
        Partitioning the graph using the Szemeredi regularity lemma.
        The algorithm used is made by Marco Fiorucci (and someone else, I will add more detailed reference)
        The partition returned is the best partition found using that k: check the power_partition function description for more details.
        
    '''
    
    sz_partition = {}
    sz_partition = power_partition(k, adj_matrix)
    
    if(len(sz_partition.keys()) == 0):
        
        return [], 0, 0
        print("Could not find any partition of the given cardinality.")
        
    else:
    
        '''
            grouping nodes by partition
        '''
        nodes_class = defaultdict(list)
        
        for i in range(len(sz_partition['classes'])):
            
            nodes_class[sz_partition['classes'][i]].append(i)
            
            
            
        '''
            Anonymizing each partition inner edges, using Erdos-Renyi model.
        '''
        
        partitions_density = {}
        partitions_edges = {}
        
        for (c, nodes) in nodes_class.items():
            partitions_density[c], partitions_edges[c] = density(nodes.copy(), adj_matrix, oriented)
            
            
        
        for (c, edges) in partitions_edges.items():
            
            adj_matrix = Erdos_Renyi(partitions_density[c], edges, adj_matrix, oriented)
            
        
        '''
            Anonymizing irregular partitions outer edges, using Erdos-Renyi model.
        '''
        
        for (g1, g2) in sz_partition['irr_list']:
            
            adj_matrix = anonymize_irr_couple(nodes_class[g1], nodes_class[g2], adj_matrix, oriented)
    
        return adj_matrix, sz_partition['epsilon'], sz_partition['sze_idx']