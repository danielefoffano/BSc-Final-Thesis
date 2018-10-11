#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:16:55 2018

@author: daniele
"""

import os
import matplotlib.pyplot as plt

def salva_plot(plot, nome_grafo, nome_file):
    os.chdir("..")
    
    print(os.getcwd())
    
    if not os.path.isdir(os.getcwd()+"/pdf_results"):
        
        os.mkdir(os.getcwd()+"/pdf_results")
    
    os.chdir("pdf_results")
    #print(os.getcwd())
    
    if not os.path.isdir(os.getcwd()+"/"+nome_grafo):
        
        os.mkdir(os.getcwd()+"/"+nome_grafo)
        
    os.chdir(nome_grafo)
    
    plot.savefig(nome_file+".pdf")
    
def scrivi_risultato(nome_grafo, euristica, frase):
    
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
    
    f = open("risultati.txt", "w+")
    f.write(frase)
    f.close()
    
    os.chdir(now)
    
#x = 3
#y = 4

#plt.plot((1, 2), (2,2))
#plt.plot(x, y, c="r", marker="o")
#salva_plot(plt, "prova", "prova")

scrivi_risultato("facebook_combined", "degree_based", "Ciao")
#plt.show()
