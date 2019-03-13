import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math, csv
import librosa
from statistics import mode
from IPython.display import Audio
from sklearn.datasets import load_iris


def template_prep(guitar=""):
    chords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A#', 'C#', 'D#', 'F#', 'G#']
    template_chords_means = {}
    template_chords = {}

    for chord in chords:
        audio, sr = librosa.load("wav_files/" + chord + guitar + ".wav", sr=None)
        # print(np.shape(librosa.feature.chroma_stft(audio, sr=sr)))
        template_chords_means[chord] = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)
        template_chords[chord] = librosa.feature.chroma_stft(audio, sr=sr)

    for chord in chords:
        audio, sr = librosa.load("wav_files/" + chord + "m" + guitar + ".wav", sr=None)
        # print(np.shape(librosa.feature.chroma_stft(audio, sr=sr)))
        template_chords_means[chord + 'm'] = np.mean(librosa.feature.chroma_stft(audio, sr=sr), axis=1)
        template_chords[chord] = librosa.feature.chroma_stft(audio, sr=sr)

    return template_chords_means, template_chords


def template_prep_combined():
    chords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A#', 'C#', 'D#', 'F#', 'G#']
    template_chords_means = {}
    template_chords = {}

    for chord in chords:
        audio1, sr = librosa.load("wav_files/" + chord + ".wav", sr=None)
        audio2, sr = librosa.load("wav_files/" + chord + "_guitar.wav", sr=None)
        template_chords[chord] = (librosa.feature.chroma_stft(audio1, sr=sr) + librosa.feature.chroma_stft(audio2, sr=sr)) / 2
        template_chords_means[chord] = np.mean(template_chords[chord], axis=1)

    for chord in chords:
        audio1, sr = librosa.load("wav_files/" + chord + "m.wav", sr=None)
        audio2, sr = librosa.load("wav_files/" + chord + "m_guitar.wav", sr=None)
        template_chords[chord] = (librosa.feature.chroma_stft(audio1, sr=sr) + librosa.feature.chroma_stft(audio2, sr=sr)) / 2
        template_chords_means[chord] = np.mean(template_chords[chord], axis=1)

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


def classify_sequence(new_sample_path, template_chords):

    # compute chromagram for each bin

    signal, sr = librosa.load(new_sample_path, sr=None)
    onset_frames = librosa.onset.onset_detect(y=signal, sr=sr, hop_length=512) * 512
    number_of_frames = onset_frames.shape[0] + 1

    chroma = np.empty((12, number_of_frames))

    index = 0
    n = 0
    for new_index in list(onset_frames):

        chroma[:, n] = np.mean(librosa.feature.chroma_stft(signal[index:new_index], sr=sr), axis=1)

        if np.all(chroma[:, n] == 0):
            chroma[:, n] = np.finfo(float).eps
        else:
            chroma[:, n] /= np.max(np.absolute(chroma[:, n]))

        n += 1
        index = new_index

    chroma[:, n] = np.mean(librosa.feature.chroma_stft(signal[index:], sr=sr), axis=1)
    n += 1

    # classify each bin

    max_classes = ["default"] * n

    for i in range(n):
        max_sim = 0.0

        for chord, template in template_chords.items():
            s = np.abs(np.inner(chroma[:,i], template) / (np.linalg.norm(chroma[:,i]) * np.linalg.norm(template)))
            if s > max_sim:
                max_sim = s
                max_classes[i] = chord

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

    print("Mean")

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

    print("Majority Vote")

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

    print("Guitar Template Data")

    tc, tc2 = template_prep("_guitar")

    print("Mean")

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

    print("Majority Vote")

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

    print("Combined Template Data")

    tc, tc2 = template_prep_combined()

    print("Mean")

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

    print("Majority Vote")

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

    print("Chord Sequence")

    tc, tc2 = template_prep()

    print(str(classify_sequence('wav_files/chords_piano_equal.wav', tc)) + " (actual)")
    print("['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C', 'Cm', 'Dm', 'Em', 'Fm', 'Gm', 'Am', 'Bm', 'Cm'] (expected)")
