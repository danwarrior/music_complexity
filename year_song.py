#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:15:35 2021

@author: jackal
"""


import pandas as pd
#%%
msd_path = ('/home/jackal/Documents/UNAM/maestría/thesis_music_complexity/mds_metadata/')

year = pd.read_csv(msd_path + 'tracks_per_year.txt')
tracks = pd.read_csv(msd_path + 'unique_tracks.txt')

#%%
y, track_id, artist, song = [], [], [], []
for i in range(len(year)):
    #print(i)
    #print(year.iloc[i,0])
    #print(year.iloc[i,0].split('<SEP>'))
    item = year.iloc[i,0].split('<SEP>')
    y.append(item[0])
    track_id.append(item[1])
    artist.append(item[2])
    song.append(item[3])
    
df_years = pd.DataFrame({'year':y, 'track_id':track_id, 'artist':artist, 'song':song})

#%%
track_id, other, artist, song = [], [], [], []
for i in range(len(tracks)):
    #print(i)
    #print(year.iloc[i,0])
    #print(year.iloc[i,0].split('<SEP>'))
    item = tracks.iloc[i,0].split('<SEP>')
    track_id.append(item[0])
    other.append(item[1])
    artist.append(item[2])
    song.append(item[3])
    
df_tracks = pd.DataFrame({'track_id':track_id, 'other':other, 'artist':artist, 'song':song})


#%%
bb_path = ('/home/jackal/Documents/UNAM/maestría/'
            'thesis_music_complexity/billboard-top-100-1950-2015/billboard/')

column_names = ['rank','artist','song']
#%%
bb_2015 = pd.read_csv(bb_path+'2015.csv')
bb_2015.columns = column_names
#%%
bb_2015['song'] = bb_2015['song'].str.lower()

