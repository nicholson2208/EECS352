import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math, csv
import librosa
from IPython.display import Audio
from sklearn.datasets import load_iris


def template_prep():
    major_chords = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    minor_chords = ['am', 'bm', 'cm', 'dm', 'em', 'fm', 'gm']

    template_chords = {}

    for chord in major_chords:
        audio, sr = librosa.load("wav_files/" + chord + "M.wav", sr=None)
        template_chords[chord] = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)

    for chord in minor_chords:
        audio, sr = librosa.load("wav_files/" + chord + "inor.wav", sr=None)
        template_chords[chord] = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)

    return template_chords


def classify(new_sample_path, template_chords):

    audio, sr = librosa.load(new_sample_path, sr=None)
    new_chroma = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)

    max_sim = 0.0
    max_class = "default"

    for chord, chroma in template_chords.items():
        s = np.abs(np.inner(new_chroma, chroma) / (np.linalg.norm(new_chroma) * np.linalg.norm(chroma)))
        if s > max_sim:
            max_sim = s
            max_class = chord

    return max_class


if __name__ == '__main__':

    tc = template_prep()

    print(classify('wav_files/AM.wav', tc))
    print(classify('wav_files/dminor.wav', tc))
    print(classify('wav_files/cminor_electric_guitar.wav', tc))
    print(classify('wav_files/DM_recorder.wav', tc))
    print(classify('wav_files/dminor_recorder.wav', tc))
    print(classify('wav_files/fminor_brass.wav', tc))
