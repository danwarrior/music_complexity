#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 11:06:52 2021

@author: jackal
"""


import numpy as np
import pandas as pd
import sklearn
import librosa
import librosa.display
import IPython.display as ipd

from sample_function import sampen_own, multiscale_sampen
import sampen # pip module

import matplotlib.pyplot as plt
import seaborn as sns
#%%
%matplotlib auto
#%%







#%% LOAD RICK
audio_path = '/home/jackal/Documents/UNAM/maestría/thesis/wav_files/rick_ashley.mp3'
rick , sr = librosa.load(audio_path)
#%%
rick_s = rick[5500:30500]
print(len(rick_s)/512)
#%%
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Numpy array of the WAV file', fontsize=25)

sample = 100
plt.subplot(121)
plt.plot(rick_s[:sample])
plt.xlabel(f'time samples (first {sample})',fontsize=15)
plt.ylabel('', fontsize=15)

plt.subplot(122)
plt.plot(rick_s)
plt.xlabel('time samples',fontsize=15)
plt.ylabel('', fontsize=15)

plt.show()
#%%
m = []
se = []
print('RICK SERIE WAV')
samp_ent = sampen.sampen2(rick_s.tolist(), mm=3, normalize=True)
for i in range(3):
    print(f'\nm={i+1}')
    m.append(i+1)
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(rick_s,
                                                                        i+1, 0.2*np.std(rick_s),'max')
    se.append(ent_mel)
    
    print(f'Entropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')

rick_entropy = pd.DataFrame({'m':m,
                             'entropy':se})













#%%
hop_length = 512
chromagram = librosa.feature.chroma_stft(rick_s, sr=sr, hop_length=hop_length)
print(chromagram.shape)

#%% CHROMA WITH LIBROSA
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
plt.show()

#%%
time_frames = chromagram.shape[1]
chroma_values = np.array([[12]*time_frames,
                   [11]*time_frames,
                   [10]*time_frames,
                   [9]*time_frames,
                   [8]*time_frames,
                   [7]*time_frames,
                   [6]*time_frames,
                   [5]*time_frames,
                   [4]*time_frames,
                   [3]*time_frames,
                   [2]*time_frames,
                   [1]*time_frames])

#%%
from scipy.stats.mstats import rankdata
chromagram_ranked = rankdata(chromagram, axis=0)
chromagram_inverse = np.flip(chromagram_ranked, axis=0) 

#%%
mask1 = np.where(chromagram_inverse==12, 1,0)
chroma_mask1 = mask1*chroma_values
serie1 = chroma_mask1.sum(axis=0)

mask2 = np.where(chromagram_inverse==11, 1,0)
chroma_mask2 = mask2*chroma_values
serie2 = chroma_mask2.sum(axis=0)

mask3 = np.where(chromagram_inverse==10, 1,0)
chroma_mask3 = mask3*chroma_values
serie3 = chroma_mask3.sum(axis=0)

#%% CHROMA MASK
plt.figure()
plt.imshow(chroma_mask1)
plt.title('Croma', fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Pitch', fontsize=20)
plt.show()

#%%
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Numpy array of the WAV file', fontsize=25)

sample = 30
plt.subplot(121)
plt.plot(serie1[:sample])
plt.xlabel(f'time samples (first {sample})',fontsize=15)
plt.ylabel('', fontsize=15)

plt.subplot(122)
plt.plot(serie1)
plt.xlabel('time samples',fontsize=15)
plt.ylabel('', fontsize=15)

plt.show()

#%%
series = pd.DataFrame({'s1':serie1,
                       's2':serie2,
                       's3':serie3})

series[:150].plot()

#%% CHROMAGRAM SERIE1 ENTROPY
# Entropy with the two functions are very similar (without normalize)
print('SERIE 1')
se1 = []
for i in range(3):
    print(f'\nm={i+1}')
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(serie1, i+1, 0.2*np.std(serie1),'max')
    se1.append(ent_mel)
    samp_ent = sampen.sampen2(serie1, mm=3, r=0.2)
    print(f'Entropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')

#%% CHROMAGRAM SERIE2 ENTROPY
print('SERIE 2')
se2 = []
for i in range(3):
    print(f'\nm={i+1}')
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(serie2, i+1, 0.2*np.std(serie2),'max')
    se2.append(ent_mel)
    samp_ent = sampen.sampen2(serie2, mm=3, r=0.2)
    print(f'Entropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')

#%% CHROMAGRAM SERIE2 ENTROPY
print('SERIE 3')
se3 = []
for i in range(2):
    print(f'm={i+1}')
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(serie3, i+1, 0.2*np.std(serie3),'max')
    se3.append(ent_mel)
    samp_ent = sampen.sampen2(serie3, mm=2, r=0.2)
    print(f'\nEntropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')

#%%
se3.append('na')
#%%

rick_entropy['se1'] = se1
rick_entropy['se2'] = se2
rick_entropy['se3'] = se3













#%%
import h5py
filename = "/home/jackal/Documents/blue_bit/music/data/msd/TRAXLZU12903D05F94.h5"

#%% GET SEGMENT PITCHES

#%%
f = h5py.File(filename,'r')
rick_pitches =  f['analysis']['segments_pitches'].value
rick_pitches = rick_pitches.T
            
#%%
time_frames = rick_pitches.shape[1]
chroma_values = np.array([[12]*time_frames,
                   [11]*time_frames,
                   [10]*time_frames,
                   [9]*time_frames,
                   [8]*time_frames,
                   [7]*time_frames,
                   [6]*time_frames,
                   [5]*time_frames,
                   [4]*time_frames,
                   [3]*time_frames,
                   [2]*time_frames,
                   [1]*time_frames])

#%%
from scipy.stats.mstats import rankdata
rick_ranked = rankdata(rick_pitches, axis=0)
#rick_inverse = np.flip(rick_ranked, axis=0) 

#%%
mask1 = np.where(rick_ranked==12, 1,0)
rick_mask1 = mask1*chroma_values
rick1 = rick_mask1.sum(axis=0)

mask2 = np.where(rick_ranked==11, 1,0)
rick_mask2 = mask2*chroma_values
rick2 = rick_mask2.sum(axis=0)

mask3 = np.where(rick_ranked==10, 1,0)
rick_mask3 = mask3*chroma_values
rick3 = rick_mask3.sum(axis=0)

#%% CHROMA MASK
plt.figure()
plt.imshow(rick_mask1[:,:100])
plt.title('Croma', fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Pitch', fontsize=20)
plt.show()
#%%
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Numpy array of the WAV file', fontsize=25)

sample = 30
plt.subplot(121)
plt.plot(rick1[:sample])
plt.xlabel(f'time samples (first {sample})',fontsize=15)
plt.ylabel('', fontsize=15)

plt.subplot(122)
plt.plot(rick1)
plt.xlabel('time samples',fontsize=15)
plt.ylabel('', fontsize=15)

plt.show()

#%%
rick_series = pd.DataFrame({'s1':rick1,
                       's2':rick2,
                       's3':rick3})

rick_series[:50].plot()

#%% CHROMAGRAM RICK1 ENTROPY
# Entropy with the two functions are very similar (without normalize)
print('SERIE 1')
msd_se1 = []
for i in range(3):
    print(f'\nm={i+1}')
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(rick1, i+1, 0.2*np.std(rick1),'max')
    msd_se1.append(ent_mel)
    samp_ent = sampen.sampen2(rick1, mm=3, r=0.2)
    print(f'Entropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')

#%% CHROMAGRAM RICK2 ENTROPY
# Entropy with the two functions are very similar (without normalize)
print('SERIE 1')
msd_se2 = []
for i in range(3):
    print(f'\nm={i+1}')
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(rick2, i+1, 0.2*np.std(rick2),'max')
    msd_se2.append(ent_mel)
    samp_ent = sampen.sampen2(rick2, mm=3, r=0.2)
    print(f'Entropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')
    
#%% CHROMAGRAM RICK3 ENTROPY
# Entropy with the two functions are very similar (without normalize)
print('SERIE 1')
msd_se3 = []
for i in range(3):
    print(f'\nm={i+1}')
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(rick3, i+1, 0.2*np.std(rick3),'max')
    msd_se3.append(ent_mel)
    samp_ent = sampen.sampen2(rick3, mm=3, r=0.2)
    print(f'Entropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')
    
#%%
msd_se3.append('na')
#%%
rick_entropy = pd.DataFrame()
rick_entropy['msd_se1'] = msd_se1
rick_entropy['msd_se2'] = msd_se2
rick_entropy['msd_se3'] = msd_se3

#%%
rick_entropy.to_csv('/home/jackal/Documents/UNAM/maestría/thesis/results/rick_entropy.csv',
                    index=False)