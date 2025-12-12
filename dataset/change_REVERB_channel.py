import os
import glob
from scipy.io import wavfile


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


files = glob.glob('REVERB/*.wav')

save_4ch = 'REVERB_4ch'
createDirectory(save_4ch)

for file in files:
    fs, signal = wavfile.read(file)

    file = file.split('/')[-1]

    file_4ch = signal[:,0:8:2]
    wavfile.write(os.path.join(save_4ch, file), fs, file_4ch)