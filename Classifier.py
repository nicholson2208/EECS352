import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math, csv
import librosa
from statistics import mode
from IPython.display import Audio
from sklearn.datasets import load_iris


def template_prep():
    chords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A#', 'C#', 'D#', 'F#', 'G#']
    template_chords_means = {}
    template_chords = {}

    for chord in chords:
        audio, sr = librosa.load("wav_files/" + chord + ".wav", sr=None)
        # print(np.shape(librosa.feature.chroma_stft(audio, sr=sr)))
        template_chords_means[chord] = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)
        template_chords[chord] = librosa.feature.chroma_stft(audio, sr=sr)

    for chord in chords:
        audio, sr = librosa.load("wav_files/" + chord + "m.wav", sr=None)
        # print(np.shape(librosa.feature.chroma_stft(audio, sr=sr)))
        template_chords_means[chord + 'm'] = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)
        template_chords[chord] = librosa.feature.chroma_stft(audio, sr=sr)

    return template_chords_means, template_chords


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


def classify2(new_sample_path, template_chords, percent_to_examine):

    audio, sr = librosa.load(new_sample_path, sr=None)
    new_chroma = librosa.feature.chroma_stft(audio, sr=sr)

    n = int(math.floor(np.shape(new_chroma)[1] * percent_to_examine))

    max_classes = ["default"] * n

    for t in range(n):
        max_sim = 0.0

        for chord, chroma in template_chords.items():
            s = np.abs(np.inner(new_chroma[:,t], chroma[:,t]) / (np.linalg.norm(new_chroma[:,t]) * np.linalg.norm(chroma[:,t])))
            if s > max_sim:
                max_sim = s
                max_classes[t] = chord

    return max_classes


def mode_classify(new_sample_path, template_chords):

    return mode(classify2(new_sample_path, template_chords, 0.75))


if __name__ == '__main__':

    tc, tc2 = template_prep()

    # print(classify('wav_files/A.wav', tc) + '   expected: A')
    # print(classify('wav_files/DM.wav', tc) + '   expected: Dm')
    # print(classify('wav_files/C_electric_guitar.wav', tc) + '   expected: C')
    # print(classify('wav_files/D_electric_guitar.wav', tc) + '   expected: D')
    # print(classify('wav_files/E_electric_guitar.wav', tc) + '   expected: E')
    #
    # print(classify('wav_files/G_synth_37th_street.wav', tc) + '   expected: G')
    # print(classify('wav_files/A_synth_37th_street.wav', tc) + '   expected: A')
    # print(classify('wav_files/B_synth_37th_street.wav', tc) + '   expected: B')
    #
    # print(classify2('wav_files/AM.wav', tc2))
    # print('expected: Am')
    # print(classify2('wav_files/D_electric_guitar.wav', tc2))
    # print('expected: D')
    # print(classify2('wav_files/E_electric_guitar.wav', tc2))
    # print('expected: E')
    # print(classify2('wav_files/G_synth_37th_street.wav', tc2))
    # print('expected: G')
    # print(classify2('wav_files/A_synth_37th_street.wav', tc2))
    # print('expected: A')

    print(classify('wav_files/A.wav', tc) + '   expected: A')
    print(classify('wav_files/DM.wav', tc) + '   expected: Dm')
    print(classify('wav_files/F#.wav', tc) + '   expected: F#')
    print(classify('wav_files/C_electric_guitar.wav', tc) + '   expected: C')
    print(classify('wav_files/D_electric_guitar.wav', tc) + '   expected: D')
    print(classify('wav_files/E_electric_guitar.wav', tc) + '   expected: E')
    print(classify('wav_files/C_recorder.wav', tc) + '   expected: C')
    print(classify('wav_files/D_recorder.wav', tc) + '   expected: D')
    print(classify('wav_files/E_recorder.wav', tc) + '   expected: E')
    print(classify('wav_files/cm_recorder.wav', tc) + '   expected: Cm')
    print(classify('wav_files/dm_recorder.wav', tc) + '   expected: Dm')
    print(classify('wav_files/em_recorder.wav', tc) + '   expected: Em')
    print(classify('wav_files/C#_recorder.wav', tc) + '   expected: C#')
    print(classify('wav_files/D#_recorder.wav', tc) + '   expected: D#')
    print(classify('wav_files/F_brassy.wav', tc) + '   expected: F')
    print(classify('wav_files/fm_brass.wav', tc) + '   expected: Fm')

    print(" ")

    print(mode_classify('wav_files/A.wav', tc2) + '   expected: A')
    print(mode_classify('wav_files/DM.wav', tc2) + '   expected: Dm')
    print(mode_classify('wav_files/F#.wav', tc2) + '   expected: F#')
    print(mode_classify('wav_files/C_electric_guitar.wav', tc2) + '   expected: C')
    print(mode_classify('wav_files/D_electric_guitar.wav', tc2) + '   expected: D')
    print(mode_classify('wav_files/E_electric_guitar.wav', tc2) + '   expected: E')
    print(mode_classify('wav_files/C_recorder.wav', tc2) + '   expected: C')
    print(mode_classify('wav_files/D_recorder.wav', tc2) + '   expected: D')
    print(mode_classify('wav_files/E_recorder.wav', tc2) + '   expected: E')
    print(mode_classify('wav_files/cm_recorder.wav', tc2) + '   expected: Cm')
    print(mode_classify('wav_files/dm_recorder.wav', tc2) + '   expected: Dm')
    print(mode_classify('wav_files/em_recorder.wav', tc2) + '   expected: Em')
    print(mode_classify('wav_files/C#_recorder.wav', tc2) + '   expected: C#')
    print(mode_classify('wav_files/D#_recorder.wav', tc2) + '   expected: D#')
    print(mode_classify('wav_files/F_brassy.wav', tc2) + '   expected: F')
    print(mode_classify('wav_files/fm_brass.wav', tc2) + '   expected: Fm')

