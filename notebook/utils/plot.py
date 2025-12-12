import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np 

def plot_melspectrogram(y, sr):
    # melspectrogram 생성
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (Noise Reduced)')
    plt.tight_layout()
    plt.show()
