"""
File containing scripts to download audio from various datasets

Also has tools to convert audio into numpy
"""
from tqdm import tqdm
import requests
import math
import os
import tarfile
import numpy as np
import librosa
import pandas as pd


import audioUtils


# ##################
# Google Speech Commands Dataset V2
# ##################

# def PrepareGoogleSpeechCmd(version=2, forceDownload=False, task='20cmd'):
def PrepareGoogleSpeechCmd(version,forceDownload=False, task='35word'):
    """
    Prepare google speech command data version2 
    tasks: Just '35word' for our project
    Returns full path to training, validation and test file list and file categories
    """
    basePath = None

    _DownloadGoogleSpeechCmdV2(forceDownload)
    basePath = 'sd_GSCmdV2'
    
    # categories in task '35word'
    if task == '35word':
        GSCmdV2Categs = {'unknown': 0,'silence': 0,'_unknown_': 0,'_silence_': 0,
            '_background_noise_': 0,'yes': 2,'no': 3,'up': 4,'down': 5,'left': 6,
            'right': 7,'on': 8,'off': 9,'stop': 10,'go': 11,'zero': 12,'one': 13,
            'two': 14,'three': 15,'four': 16,'five': 17,'six': 18,'seven': 19,
            'eight': 20,'nine': 1,'backward': 21,'bed': 22,'bird': 23,'cat': 24,
            'dog': 25,'follow': 26,'forward': 27,'happy': 28,'house': 29,
            'learn': 30,'marvin': 31,'sheila': 32,'tree': 33,'visual': 34,'wow': 35}
        numGSCmdV2Categs = 36


    print('Converting test WAVs to numpy files, Data augmentation for test set')
    test_aug = audioUtils.WAV2Numpy(basePath + '/test/')
    print('Converting training set WAVs to numpy files, Data augmentation for train set')
    train_aug = audioUtils.WAV2Numpy(basePath + '/train/')

    # read split from files and all files in folders
    testing = pd.read_csv(basePath + '/train/testing_list.txt',
                           sep=" ", header=None)[0].tolist()
    validation = pd.read_csv(basePath + '/train/validation_list.txt',
                          sep=" ", header=None)[0].tolist()

    testing = [os.path.join(basePath + '/train/', f + '.npy')
                for f in testing if f.endswith('.wav')]
    validation = [os.path.join(basePath + '/train/', f + '.npy')
               for f in validation if f.endswith('.wav')]
    
    if test_aug is not None:
        testing.extend(test_aug) # add augmentated test files name
    
    
    allWAVs = []
    for Path, dirs, files in os.walk(basePath + '/train/'):
        allWAVs += [Path + '/' + f for f in files if f.endswith('.wav.npy')]
    
    if train_aug is not None:
        allWAVs.extend(train_aug) #add augmentated train files name
    training = list(set(allWAVs) - set(validation) - set(testing))

    testWAVsREAL = []
    for Path, dirs, files in os.walk(basePath + '/test/'):
        testWAVsREAL += [Path + '/' +
                         f for f in files if f.endswith('.wav.npy')]

    # get categories
    testing_label = [_getFileCategory(f, GSCmdV2Categs) for f in testing]
    validation_label = [_getFileCategory(f, GSCmdV2Categs) for f in validation]
    training_label = [_getFileCategory(f, GSCmdV2Categs) for f in training]
    testWAVREALlabels = [_getFileCategory(f, GSCmdV2Categs)
                         for f in testWAVsREAL]

    # use background noise as validation
    BN_list = [training[i] for i in range(len(training_label))
                      if training_label[i] == GSCmdV2Categs['silence']]
    BN_categ = [GSCmdV2Categs['silence']
                     for i in range(len(BN_list))]
    if numGSCmdV2Categs == 12:
        validation += BN_list
        validation_label += BN_categ

    # build dictionaries
    testing_dict = dict(zip(testing, testing_label))
    validation_dict = dict(zip(validation, validation_label))
    training_dict = dict(zip(training, training_label))
    testing_dict_real = dict(zip(testWAVsREAL, testWAVREALlabels))

    trainInfo = {'files': training, 'labels': training_dict}
    valInfo = {'files': validation, 'labels': validation_dict}
    testInfo = {'files': testing, 'labels': testing_dict}
    testREALInfo = {'files': testWAVsREAL, 'labels': testing_dict_real}
    gscInfo = {'train': trainInfo,
               'test': testInfo,
               'val': valInfo,
               'testREAL': testREALInfo}

    print('Done preparing Google Speech commands dataset version {}'.format(version))

    return gscInfo, numGSCmdV2Categs


def _getFileCategory(file, catDict):
    """
    Receives a file with name sd_GSCmdV2/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ, 0)


def _DownloadGoogleSpeechCmdV2(forceDownload=False):
    """
    Downloads google speech commands data version 2
    """
    if os.path.isdir("sd_GSCmdV2/") and not forceDownload:
        print('Google Speech commands dataset version 2 already exists. Skipping download.')
    else:
        if not os.path.exists("sd_GSCmdV2/"):
            os.makedirs("sd_GSCmdV2/")
        trainFiles = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        testFiles = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
        _downloadFile(testFiles, 'sd_GSCmdV2/test.tar.gz')
        _downloadFile(trainFiles, 'sd_GSCmdV2/train.tar.gz')

    # extract files
    if not os.path.isdir("sd_GSCmdV2/test/"):
        _extractTar('sd_GSCmdV2/test.tar.gz', 'sd_GSCmdV2/test/')

    if not os.path.isdir("sd_GSCmdV2/train/"):
        _extractTar('sd_GSCmdV2/train.tar.gz', 'sd_GSCmdV2/train/')


##############
# Utilities
##############


def _downloadFile(url, fName):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    batch_size = 1024
    wrote = 0
    print('Downloading {} into {}'.format(url, fName))
    with open(fName, 'wb') as f:
        for data in tqdm(r.iter_content(batch_size),
                         total=math.ceil(total_size // batch_size),
                         unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)


def _extractTar(fname, folder):
    print('Extracting {} into {}'.format(fname, folder))
    if (fname.endswith("tar.gz")):
        file = tarfile.open(fname, "r:gz")
        file.extractall(path=folder)
        file.close()
    elif (fname.endswith("tar")):
        file = tarfile.open(fname, "r:")
        file.extractall(path=folder)
        file.close()
