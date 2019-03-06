import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math, csv
import librosa
from IPython.display import Audio
from sklearn.datasets import load_iris

major_chords = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
minor_chords = ['am', 'bm', 'cm', 'dm', 'em', 'fm', 'gm']

template_chords = {}

for chord in major_chords:
    audio, sr = librosa.load("wav_files/" + chord + "M.wav", sr=None)
    template_chords[chord] = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)

for chord in minor_chords:
    audio, sr = librosa.load("wav_files/" + chord + "inor.wav", sr=None)
    template_chords[chord] = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)
