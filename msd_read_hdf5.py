#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:18:25 2020

@author: jackal
"""

import numpy as np
import h5py
filename = "/home/jackal/Documents/blue_bit/music/data/msd/TRAXLZU12903D05F94.h5"

#%%
f = h5py.File(filename,'r')
for i in  f.keys(): # Names of the groups in HDF5 file.
    print('\nGROUP',i)
    for j in f[i]: # Checkout which items are inside each group.
        print(' -item   ----->',j)
        print('  ',type(f[i][j].value),f[i][j].value.shape)
        #if i == 'analysis' and j == 'segments_pitches':
        #if i == 'musicbrainz' and j == 'songs':
        if i == 'metadata' and j == 'artist_terms': #GENRES
        #if i == 'metadata' and j == 'songs':
            print(f[i][j].value) # Get values for specific GRUOP-ITEM 
            #data = f[i][j].value
            #data = [k for k in f[i][j].value[0]]
# After you are done
#f.close()

#data_np = np.array([k for k in data.value[0]])
#%% - EXTRACTOR
f = h5py.File(filename,'r')
for i in  f.keys(): # Names of the groups in HDF5 file.
    print('\nGROUP',i)
    for j in f[i]: # Checkout which items are inside each group.
        #print(' -item   ----->',j)
        #print('  ',type(f[i][j].value),f[i][j].value.shape)
        #if i == 'analysis' and j == 'songs':
        #if i == 'musicbrainz' and j == 'songs':
        #if i == 'metadata' and j == 'artist_terms':
        if i == 'metadata' and j == 'songs': # NAME OF THE SONG
            print('  ',type(f[i][j].value),f[i][j].value.shape)
            print(f[i][j].value)
            data = f[i][j].value
#%%            
        if i == 'metadata' and j == 'artist_terms': # GENRES
            print('  ',type(f[i][j].value),f[i][j].value.shape)
            print(f[i][j].value)
            data = f[i][j].value     


#%%
data = data.T

#%%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

#%%
%matplotlib auto
#%% LIBROSA CHROMOGRAM
hop_length = 512
plt.figure(figsize=(15, 5))
librosa.display.specshow(data, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
plt.show()

#%% PYTHON CROMOGRAM
plt.figure()
plt.imshow(data[:,:50])
plt.title('Croma', fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Pitch', fontsize=20)
plt.show()

#%%
import scipy.stats as ss
data_ranked_vec = ss.rankdata(data[:,0])
#%%
from scipy.stats.mstats import rankdata
data_ranked = rankdata(data, axis=0)

#%%
sample = data[:,:30]
#%%
masked = np.where(data==1, 1,0)
#%% CHROMA VALUES
chroma_values = np.array([[12]*935,
                   [11]*935,
                   [10]*935,
                   [9]*935,
                   [8]*935,
                   [7]*935,
                   [6]*935,
                   [5]*935,
                   [4]*935,
                   [3]*935,
                   [2]*935,
                   [1]*935,])

#%%
chroma_mask = masked*chroma_values
#%%
serie = chroma_mask.sum(axis=0)
#%%
plt.figure()
plt.imshow(data[:,:50])
plt.title('Croma', fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Pitch', fontsize=20)
plt.show()

#%%
#%%
plt.figure()
plt.imshow(masked)
plt.title('Croma', fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Pitch', fontsize=20)
plt.show()
#%%
plt.figure()
plt.plot(serie[:100])
plt.title('Croma', fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Pitch', fontsize=20)
plt.show()










