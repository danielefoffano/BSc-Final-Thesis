#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:28:10 2018

@author: daniele
"""

import sys

if './regularity_lemma/' not in sys.path:
    sys.path.insert(1, './regularity_lemma/')

import szemeredi_lemma_builder as sz

'''
    Function to find a Szemeredi regular partition with a cardinality equal to the k inserted by the user.
    The k must be a power of two.
    Multiple partitions are examined, using different epsilons within a range going from 0.25 to 0.01.
    The chosen partition will be the best partition found for that k, comparing the epsilon of each partition found (the smallest, the best).

    Function parameters:
        * user_k : the k inserted by the user
        * adj_matrix : the adjacency matrix of the graph
    
    Function output: 
        * a dictionary with the following structure:
            {
                'k' : the number of partitions
                'epsilon' : the epsilon of the partition
                'classes' : array of size n (number of nodes in the graph); the node i belongs to the class classes[i] 
                'sze_idx' : the szemeredy_index used to evaluate the quality of the partition
                'irr_pairs' : number of irregular pairs
                'irr_list' : a list of the irregular pairs
            }
'''

def power_partition ( user_k, adj_matrix):
    
    if( user_k == 0 or not(user_k & (user_k-1) == 0)): # Check if k is not power of 2
        
        print("Devi inserire un k che sia potenza di 2")
        return {}
    
    else:    
        #print("Bravo hai inserito una potenza di 2")
        
        epsilon_range = [0.5, 0.01]
        
        partitions = {}
        
        while(epsilon_range[0] > epsilon_range[1]):
            #print("provo con epsilon {}".format(epsilon_range[0]))
            alg = sz.generate_szemeredi_reg_lemma_implementation('alon', adj_matrix, epsilon_range[0], False, True, 'degree_based', False)
            
            is_regular, k, classes, sze_idx, regularity_list, irr_pairs, irr_list = alg.run()
            
            if is_regular: 
                
                if k not in partitions.keys(): # If this is the first partion with cardinality k, store it
                    
                    partitions[k] = {
                                        'k' : k,
                                        'epsilon' : epsilon_range[0],
                                        'classes' : classes,
                                        'sze_idx' : sze_idx,
                                        'irr_pairs' : irr_pairs,
                                        'irr_list' : irr_list
                                    }
                else:
                    
                    # Not the first partition with cardinality k: substitute prev one if this has a better epsilon
                    if partitions[k]['epsilon'] > epsilon_range[0] and partitions[k]['irr_pairs'] >= irr_pairs: 
                        
                        partitions[k] = {
                                        'k' : k,
                                        'epsilon' : epsilon_range[0],
                                        'classes' : classes,
                                        'sze_idx' : sze_idx,
                                        'irr_pairs' : irr_pairs,
                                        'irr_list' : irr_list
                                    }
            
            epsilon_range[0] = epsilon_range[0] - 0.01
            
        if (user_k in partitions.keys()):
            
            return partitions[user_k]
        
        else:
        
            return {}
            
        