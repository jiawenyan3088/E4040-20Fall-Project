"""
Utility functions for audio files
"""
import librosa
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.io import wavfile

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype(np.float) / np.sum(cm, axis=1)
        cm = cm[:, np.newaxis]
        print("Done, Normalization of Confusion Matrix")
    else:
        print("No Normalization")

    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x = list(range(0, len(classes), 1))
    plt.xticks(x, classes, rotation=45, fontsize=12)
    plt.yticks(x, classes, fontsize=12)

    if normalize:
        fmt = '.3f'
    else:
        fmt = 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        Color = " "
        if (cm[i, j] > thresh):
            Color = "white"
        else:
            Color = "black"
        plt.text(j, i, format(cm[i, j], fmt), size=11, horizontalalignment="center", color=Color)

    plt.ylabel('True Label', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=30)
    plt.savefig('picConfMatrix.png', dpi=400)
    plt.tight_layout()


def WAV2Numpy(folder, sr=None):
    """
    Convert Wav files into Numpy Arrays
    """
    #Add File names into a list
    Params = {"NJ_mode": True, "noise_factor": 0.01,  # parameters for noise injection
          "ST_mode":True, "shift_length": 1000, "shift_direction": "right",  # parameters for shifting time
          "CP_mode":True, "sr": 44, "n_step": 0,  # parameters for changing pitch
          "CS_mode":True, "rate": 1.01,  # parameters for changing speed
          "sampling_rate": 0.2  # proportion of sample to apply augmentation
         }

    allFiles = []
    for root, dirs, files in os.walk(folder):
        allFiles += [os.path.join(root, f) for f in files
                     if f.endswith('.wav')]
    aug_list = []
    for file in tqdm(allFiles):
        y, sr = librosa.load(file, sr=None)

        # if we want to write the file later
        # librosa.output.write_wav('file.wav', y, sr, norm=False)

        if random.uniform(0,1) <= Params['sampling_rate']:
            y_aug = Data_Augmentation.noise_injection(y,Params['noise_factor'])
            y_aug = Data_Augmentation.shifting_time(y,Params['shift_length'],Params['shift_direction'])
#         if random.uniform(0,1) <= 0.2:
#             y_aug = DataAug.noise_injection(y,0.01)
#             y_aug = DataAug.shifting_time(y,1000,'right')

            aug_filename = file[:-4] + "_aug" + file[-4:] + '.npy'
            aug_list.append(aug_filename)
            np.save(aug_filename,y_aug)
        
        np.save(file + '.npy', y)
        os.remove(file)
    
