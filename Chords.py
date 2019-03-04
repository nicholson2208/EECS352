# This line is a convenience to import most packages you'll need. You may need to import others (e.g. random and cmath)
import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math, csv
import librosa
from IPython.display import Audio
from sklearn.datasets import load_iris


audio = librosa.load("c_chord.wav", sr=None)[0]
Audio(audio, rate=44100)
print("done")