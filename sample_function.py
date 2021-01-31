#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:00:25 2021

@author: jackal
"""
import numpy as np

def sampen_own(L, m, r, distance):
    N = len(L)
    B = 0.0
    A = 0.0
    
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])
    
    # En este paso se usa la distancia del máximo
    if distance == 'max':
        matches_b = [np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi]
    else:
        matches_b = [np.sum(np.linalg.norm(xmii - xmj) <= r) - 1 for xmii in xmi] # not working
    # Save all matches minus the self-match, compute B
    B = np.sum(matches_b)

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])
    
    if distance == 'max':
        matches_a = [np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm]
    else:
        matches_a = [np.sum(np.linalg.norm(xmi - xmj) <= r) - 1 for xmi in xm] # not working

    A = np.sum(matches_a)

    # Return SampEn
    #return -np.log(A / B)
    return -np.log(A / B), A, B, xmi, xmj, matches_b, xm



def multiscale_sampen(S, m, r, delta=1):
    N = len(S)
    B = 0.0
    A = 0.0
    
    # delta reduction process
    delta_indexes = [e*delta for e in range(int(np.floor(N/delta)))]
    delta_reduced_serie = S[tuple([delta_indexes])]
    S = delta_reduced_serie
    N = len(S)
    
    
    # Split time series and save all templates of length m
    xmi = np.array([S[i : i + m] for i in range(N - m)])
    xmj = np.array([S[i : i + m] for i in range(N - m + 1)])
    
    # En este paso se usa la distancia del máximo
    matches_b = [np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi]

    # Save all matches minus the self-match, compute B
    B = np.sum(matches_b)

    # Similar for computing A
    m += 1
    xm = np.array([S[i : i + m] for i in range(N - m + 1)])
    
    matches_a = [np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm]
    
    A = np.sum(matches_a)

    # Return SampEn
    #return -np.log(A / B)
    return -np.log(A / B), A, B, xmi, xmj, matches_b, xm

#%%
    
#entropy, A, B, xmi, xmj, matches, xm = sampen(serie, 2, 0.9, 'max')
#print(entropy)
#%%
#for i in xmi[:5]:

#    print('-\n-->',i)
#    print((xmj-i)[:5])
#    print(np.abs(xmj - i)[:5])
#    print(np.abs(xmj - i)[:5].max(axis=1))
##    print(np.abs(xmj - i).max(axis=1) <= 0.9)
#    print(np.sum(np.abs(xmj - i).max(axis=1) <= 0.9))
    
#%%
#v = np.array([11,4])
#u = np.array([7,2])
#print(np.linalg.norm(u - v))

#%%
#print(u - v)
#rint((u-v)**2) 
#print(np.sum((u-v)**2))
#print(np.sqrt(np.sum((u-v)**2))) 