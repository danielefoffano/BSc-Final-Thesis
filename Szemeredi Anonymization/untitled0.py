#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:43:23 2018

@author: daniele
"""
import operator

a = {1 : 0, 2: 3, 3 : 1}

print(list(a.items()))

b = list(a.items())

b.sort(key = operator.itemgetter(1), reverse = True)

b = [x for (x,y) in b]
print(b)