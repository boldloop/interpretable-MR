import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import random

random.seed(5672349)

audio_dir = "GTZAN/genres_original"

def make_or_clean_dir(d):
    try:
        os.mkdir(d)
    except FileExistsError:
        shutil.rmtree(d)
        os.mkdir(d)


def write_spec_from_wav(wav_path, spec_base):
    signal, sr = librosa.load(wav_path)
    spec = librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=sr),
                               ref=np.max)
    fig, ax = plt.subplots()
    width = spec.shape[1] // 10
    for i in range(10):
        librosa.display.specshow(spec[:, i*width:(i+1)*width],
                                 ax=ax, cmap='gray_r')
        fig.savefig(spec_base + str(i) + '.png')
    plt.close(fig)


data_folders = ["split_data/train", "split_data/test"]
make_or_clean_dir("split_data")
for d in data_folders:
    make_or_clean_dir(d)

for genre in os.listdir(audio_dir):
    print(f'starting {genre}')
    wav_names = list(os.listdir(os.path.join(audio_dir, genre)))
    random.shuffle(wav_names)
    test_paths = wav_names[:20]
    train_paths = wav_names[20:]

    for d in data_folders:
        make_or_clean_dir(os.path.join(d, genre))

    for wave_name in test_paths:
        write_spec_from_wav(
            os.path.join(audio_dir, genre, wave_name),
            os.path.join(data_folders[1], genre, wave_name.replace('.wav', '')),
        )
    print('finished test')

    for wave_name in train_paths:
        write_spec_from_wav(
            os.path.join(audio_dir, genre, wave_name),
            os.path.join(data_folders[0], genre, wave_name.replace('.wav', '')),
        )
    print('finished train')

# done.
