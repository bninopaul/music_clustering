import os
import re
from itertools import chain
from pydub import AudioSegment
from pydub.utils import mediainfo
import numpy as np
import pandas as pd

def cameltonormal(string):
    new = re.sub('(.)([A-Z])', r'\1 \2', string)
    return new[0].upper()+new[1:]
def get_stringdata(fil):
    "Extracts the song title and artist"
    splitted = fil.split("_")
    artist = cameltonormal(splitted[0])
    song = cameltonormal(splitted[1])
    return artist, song
def get_numericdata(directory):
    audio = AudioSegment.from_mp3(directory)
    audio_bytestring = audio._data
    audio_signal = np.fromstring(audio_bytestring, dtype=np.int32()).astype(np.int16())

    info = mediainfo(directory)
    sample_rate = int(info['sample_rate'])
    channels = int(info['channels'])
    return audio_signal, sample_rate, channels
def process_music(dir_):
    dct = {}
    for i in os.listdir(dir_):
        filei = os.path.join(dir_, i)
        if os.path.isfile(filei) and re.match(".*.mp3", i):
            string = get_stringdata(re.sub(".mp3", "", i).strip())
            numeric = get_numericdata(filei)
            if string[0] in dct:
                dct[string[0]].append((string[1],numeric))
            else:
                dct[string[0]] = [(string[1],numeric)]
    df = pd.DataFrame(list(chain.from_iterable((((k, v) for v in vals) for (k, vals) in dct.items()))),
                  columns=('artist', 'data'))
    df['song'] = df['data'].map(lambda x: x[0])
    df['signal'] = df['data'].map(lambda x: x[1][0])
    df['sample_rate'] = df['data'].map(lambda x: x[1][1])
    df['channel'] = df['data'].map(lambda x: x[1][2])
    df.drop('data', 1,inplace = True)
    return df
