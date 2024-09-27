import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os

# Create 'songs' directory if it doesn't exist
if not os.path.exists('songs'):
    os.makedirs('songs')


auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)


def getTrackIDs(user, playlist_id):
    track_ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        track_ids.append(track['id'])
    return track_ids

def getTrackFeatures(id):
    track_info = sp.track(id)

    name = track_info['name']
    album = track_info['album']['name']
    artist = track_info['album']['artists'][0]['name']
    release_date = track_info['album']['release_date']
    length = track_info['duration_ms']
    popularity = track_info['popularity']

    track_data = [name, album, artist] #, release_date, length, popularity
    return track_data

# Code for creating dataframe of feteched playlist

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist = {
    0: "0l9dAmBrUJLylii66JOsHB",
    1: "1n6cpWo9ant4WguEo91KZh",
    2: "4cllEPvFdoX6NIVWPKai9I",
    3: "0deORnapZgrxFY4nsKr9JA",
    4: "4kvSlabrnfRCQWfN0MgtgA",
    5: "1n6cpWo9ant4WguEo91KZh",
    6: "37i9dQZEVXbMDoHDwVN2tF"
}


'''
Code can def be modularised into a function but i tried to write it when i was extremely sleepy so thought screw it and repeated code block

Uncomment for fetching updated playlists
'''


track_ids = getTrackIDs('spotify',music_dist[0])
track_list = []
for i in range(len(track_ids)):
    time.sleep(.3)
    track_data = getTrackFeatures(track_ids[i])
    track_list.append(track_data)
    df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
    df.to_csv('songs/angry.csv')
print("CSV Generated")
track_ids = getTrackIDs('spotify',music_dist[1])
track_list = []
for i in range(len(track_ids)):
    time.sleep(.3)
    track_data = getTrackFeatures(track_ids[i])
    track_list.append(track_data)
    df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
    df.to_csv('songs/disgusted.csv')
print("CSV Generated")
track_ids = getTrackIDs('spotify',music_dist[2])
track_list = []
for i in range(len(track_ids)):
    time.sleep(.3)
    track_data = getTrackFeatures(track_ids[i])
    track_list.append(track_data)
    df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
    df.to_csv('songs/fearful.csv')
print("CSV Generated")
track_ids = getTrackIDs('spotify',music_dist[3])
track_list = []
for i in range(len(track_ids)):
    time.sleep(.3)
    track_data = getTrackFeatures(track_ids[i])
    track_list.append(track_data)
    df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
    df.to_csv('songs/happy.csv')
print("CSV Generated")

track_ids = getTrackIDs('spotify',music_dist[4])
track_list = []
for i in range(len(track_ids)):
    time.sleep(.3)
    track_data = getTrackFeatures(track_ids[i])
    track_list.append(track_data)
    df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
    df.to_csv('songs/neutral.csv')
print("CSV Generated")
track_ids = getTrackIDs('spotify',music_dist[5])
track_list = []
for i in range(len(track_ids)):
    time.sleep(.3)
    track_data = getTrackFeatures(track_ids[i])
    track_list.append(track_data)
    df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
    df.to_csv('songs/sad.csv')
print("CSV Generated")
track_ids = getTrackIDs('spotify',music_dist[6])
track_list = []
for i in range(len(track_ids)):
    time.sleep(.3)
    track_data = getTrackFeatures(track_ids[i])
    track_list.append(track_data)
    df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
    df.to_csv('songs/surprised.csv')
print("CSV Generated")