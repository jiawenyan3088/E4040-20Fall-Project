import numpy as np
import tensorflow.keras


class AudioGen(tensorflow.keras.utils.Sequence):
    """
    Generator that generate batches of audio samples for training
    
    list_IDs: File names that in a list
    lables: Their corresponding labels
    They should be the same length
    """
    def __init__(self, list_IDs, labels, batch_size=32,
                 dim=16000, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Calculate the overall batch numbers'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
    def on_epoch_end(self):
        'index updation'
        self.indexes = np.arange(len(self.list_IDs))
        #shuffle the batch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # According indexes, find the cooresponding file names 
        list_IDs_temp = [self.list_IDs[i] for i in indexes]

        # Data generation
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty((self.batch_size, self.dim),dtype=float)
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):            
            cur = np.load(ID)
            #Number of audio samples in column
            #Store the audio samples
            if cur.shape[0] == self.dim:
                x[i] = cur
            #If the the length of the file is bigger than 16000
            #Then we randomly choose 16000 consecutive samples in this audio file
            #To make sure the total length always equal to 16000
            elif cur.shape[0] > self.dim:  
                exceed_len = cur.shape[0] - self.dim
                #Random Choice
                start_idx = np.random.randint(exceed_len)
                x[i] = cur[start_idx:start_idx+self.dim]
            #If smaller, we need to randomly choose a start point in x
            #And then copy all the  
            #And then leave the rest as 0
            else:
                exceed_len = self.dim - cur.shape[0]
                #Random Choice
                start_idx = np.random.randint(exceed_len)
                x[i, start_idx:start_idx + cur.shape[0]] = cur

            # Store the class
            y[i] = self.labels[ID]

        return x, y
