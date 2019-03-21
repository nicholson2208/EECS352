import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math, csv
import librosa
from statistics import mode
from IPython.display import Audio
from sklearn.datasets import load_iris
import utilities as util


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
        audio, sr = librosa.load("wav_files/" + chord.lower() + "m" + guitar + ".wav", sr=None)
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
        audio1, sr = librosa.load("wav_files/" + chord.lower() + "m.wav", sr=None)
        audio2, sr = librosa.load("wav_files/" + chord.lower() + "m_guitar.wav", sr=None)
        template_chords[chord] = (librosa.feature.chroma_stft(audio1, sr=sr) + librosa.feature.chroma_stft(audio2, sr=sr)) / 2
        template_chords_means[chord] = np.mean(template_chords[chord], axis=1)

    return template_chords_means, template_chords


def mean_classify(new_sample_path, template_chords):

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


def frame_classify(new_sample_path, template_chords, percent_to_examine):

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

    return mode(frame_classify(new_sample_path, template_chords, 0.75))


def accuracy(expected, classify, tc):
    correct = 0
    n = len(expected)
    actual = ["default"] * n
    for i in range(n):
        new = classify("wav_files/" + str(i) + ".wav", tc)
        if new == expected[i]:
            correct += 1
        actual[i] = new
    return correct / n, actual


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


if __name__ == '__main__':

    mapping = util.create_wav_file_mapping("wav_file_mapping.txt")
    expected = util.get_chord_name(mapping, 24)[0]
    tc_means, tc_frames = template_prep()

    print("Mean")

    print(mean_classify('wav_files/A.wav', tc_means) + '   expected: A')
    print(mean_classify('wav_files/DM.wav', tc_means) + '   expected: Dm')
    print(mean_classify('wav_files/F#.wav', tc_means) + '   expected: F#')
    mean_results = accuracy(expected, mean_classify, tc_means)
    print(mean_results)

    print("Majority Vote")

    print(mode_classify('wav_files/A.wav', tc_frames) + '   expected: A')
    print(mode_classify('wav_files/DM.wav', tc_frames) + '   expected: Dm')
    print(mode_classify('wav_files/F#.wav', tc_frames) + '   expected: F#')
    mode_results = accuracy(expected, mode_classify)

    print("Chord Sequence")

    print(str(classify_sequence('wav_files/chords_piano_equal.wav', tc_means)) + " (actual)")
    print("['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C', 'Cm', 'Dm', 'Em', 'Fm', 'Gm', 'Am', 'Bm', 'Cm'] (expected)")
