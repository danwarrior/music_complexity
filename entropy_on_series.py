#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:05:18 2021

@author: jackal
"""


import numpy as np
import pandas as pd
import sklearn
import librosa
import IPython.display as ipd

from sample_function import sampen_own, multiscale_sampen
import sampen # pip module


#%% - ENTROPY ON MY SERIES
entropy = sampen(serie, 2, 0.9, 'max')
print(entropy)









#%% LOAD RICK
audio_path = '/home/jackal/Documents/UNAM/maestría/thesis/wav_files/rick_ashley.mp3'
#ipd.Audio(audio_path)

rick , sr = librosa.load(audio_path)
#%%
print(len(rick))
print(len(rick)/935)
#%%
fig = plt.figure(figsize=(15, 5))
fig.suptitle('Numpy array of the WAV file', fontsize=25)

sample = 10000
plt.subplot(121)
plt.plot(rick[:sample])
plt.xlabel(f'time samples (first {sample})',fontsize=15)
plt.ylabel('', fontsize=15)

plt.subplot(122)
plt.plot(rick)
plt.xlabel('time samples',fontsize=15)
plt.ylabel('', fontsize=15)

plt.show()

#%%
r_rick = 0.2*np.std(rick)
print(r_rick)
#%% RICK ENTROPY
rick_ent,_,_,_,_,_,_ = sampen_own(rick[:5000], 2, r_rick,'max')
print(rick_ent)

rick_ent = sampen.sampen2(rick[:5000].tolist(), mm=2, normalize=True)
print(rick_ent[2])
#%%
print('RICK SERIE WAV')
for i in range(3):
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(rick[:5000],
                                                                        i+1, 0.2*np.std(rick[:5000]),'max')
    samp_ent = sampen.sampen2(rick[:5000].tolist(), mm=3, normalize=True)
    print(f'\nEntropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')














#%% RECORD EXAMPLE MELODICA
audio_path = '/home/jackal/Documents/UNAM/maestría/thesis_music_complexity/wav_files/2021-01-23-12:40:16.wav'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))
print(f'Shape :{x.shape}, Sample rate :{sr}')
print(f'mean:{np.mean(x):1f}, min:{np.min(x):1f}, max:{np.max(x):1f}, std:{np.std(x):1f}')
#%% SET r PARAMETER
r = 0.2*np.std(x)
print(r)
#%% SampEn FULL AMPLITUDE SERIES ---> THEY MATCH!!!
print('FULL SERIE WAV')
for i in range(3):
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(x[:5000],
                                                                        i+1, 0.2*np.std(x[:5000]),'max')
    samp_ent = sampen.sampen2(x[:5000].tolist(), mm=3, normalize=True)
    print(f'\nEntropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')
#%%
print(f'A: {mel_A}')
print(f'B: {mel_B}')
print(mel_i[:5])
print(f'mathces: {len(mel_match)}')
print(f'mathces: {mel_match[:5]}')
#%%
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
print(chromagram.shape)

chromagram_inverse = np.flip(chromagram, axis=0) 
#%% CHROMA WITH LIBROSA
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram[:,:500], x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
plt.show()
#%% CHROMA WITH PLT
plt.figure(figsize=(15, 5))
plt.imshow(chromagram[:,:500])
plt.title('Croma with PLT', fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Pitch', fontsize=20)
plt.show()
#%% CHROMA INVERSE
# This inverted chroma emultales the LIBROSA plot
plt.figure()
plt.imshow(chromagram_inverse[:,:500])
plt.title('Croma (inverted) with PLT', fontsize=25)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Pitch', fontsize=20)
plt.show()

#%% ONES SERIES
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
plt.imshow(chroma_mask1[:,:500])
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
for i in range(3):
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(serie1, i+1, 0.2*np.std(serie1),'max')
    samp_ent = sampen.sampen2(serie1, mm=3, r=0.2)
    print(f'\nEntropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')
#%%
print(f'A: {mel_A}')
print(f'B: {mel_B}')
print(mel_i[:5])
print(mel_j[:5])
print(mel_match[:5])

#%% CHROMAGRAM SERIE2 ENTROPY
print('SERIE 2')
for i in range(5):
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(serie2, i+1, 0.2*np.std(serie2),'max')
    samp_ent = sampen.sampen2(serie2, mm=5, r=0.2)
    print(f'\nEntropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')
#%%
print(f'A: {mel_A}')
print(f'B: {mel_B}')
print(mel_i[:5])
print(mel_j[:5])
print(mel_match[:5])

#%% CHROMAGRAM SERIE2 ENTROPY
print('SERIE 3')
for i in range(5):
    ent_mel, mel_A, mel_B, mel_i, mel_j, mel_match, mel_xm = sampen_own(serie3, i+1, 0.2*np.std(serie3),'max')
    samp_ent = sampen.sampen2(serie3, mm=5, r=0.2)
    print(f'\nEntropy: {ent_mel}')
    print(f'Entropy: {samp_ent[i+1][1]}')
#%%
print(f'A: {mel_A}')
print(f'B: {mel_B}')
print(mel_i[:5])
print(mel_j[:5])
print(mel_match[:5])












#%% TESTS ON MULTISCALE

print('SERIE 1')
for i in range(3):
    e_ent, e_A, e_B, e_i, e_j, e_match_b, e_xm = sampen_own(serie1, i+1, 0.2*np.std(serie1),'max')
    samp_ent = sampen.sampen2(serie1, mm=3, r=0.2)
    me_ent, me_A, me_B, me_i, me_j, me_match, me_xm = multiscale_sampen(serie1,
                                                                               i+1, 0.2*np.std(serie1), 2)
    print(f'\nEntropy: {e_ent}')
    print(f'Entropy: {samp_ent[i+1][1]}')
    print(f'MultiEntropy: {me_ent}')
#%%
for i in range(3):
    me_ent, me_A, me_B, me_i, me_j, me_match, me_xm = multiscale_sampen(serie1,
                                                                               i+1, 0.2*np.std(serie1))
    print(f'\nMultiEntropy: {me_ent}')
#%%
print(f'A: {mel_A}')
print(f'B: {mel_B}')
print(mel_i[:5])
print(mel_j[:5])
print(mel_match[:5])