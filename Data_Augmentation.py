import numpy as np
import random
import librosa

def noise_injection(data, noise_factor):
    noise = np.random.randn(len(data))
    injected = data + noise_factor * noise
    # Cast back to same data type
    injected = injected.astype(type(data[0]))
    return injected

def shifting_time(data, shift_length, shift_direction):
    shift = np.random.randint(shift_length)
    if shift_direction == 'right':
        shift = -shift
    shifted = np.roll(data, shift)

    if shift > 0:
        shifted[:shift] = 0
    else:
        shifted[shift:] = 0
    return shifted

def changing_pitch(data,sr, n_step):
    """
    params: 
        sr: sampling rate
        n_step: number of steps to change pitch, can be pos/neg
    """
    pitch_changed = librosa.effects.pitch_shift(data,sr,n_step)
    return pitch_changed

def changing_speed(data, rate):
    """
    params:
        rate: speed up if rate>1,slow down if rate <1
    """
    speed_changed = librosa.effects.time_stretch(data, rate)
    return speed_changed

def batch_aug(x_data,y_class,params):

    if params["sampling_rate"] > 1:
        print("Sampling rate must be less than 1")
    data_size = len(x_data)
    sampling_size = int(data_size * params["sampling_rate"])

    if params['NJ_mode']:
        index = random.sample(range(0,data_size), sampling_size)
        injected = np.apply_along_axis(noise_injection,1,x_data[index],params['noise_factor'])

        x_data = np.vstack((x_data,injected))
        y_class = np.hstack((y_class,y_class[index]))

    if params['ST_mode']:
        index = random.sample(range(0,data_size), sampling_size)
        shifted = np.apply_along_axis(shifting_time,1,x_data[index],params['shift_length'],params['shift_direction'])

        x_data = np.vstack((x_data,shifted))
        y_class = np.hstack((y_class,y_class[index]))

    if params['CP_mode']:
        try:
            index = random.sample(range(0,data_size), sampling_size)
            pitch_changed = np.apply_along_axis(changing_pitch,1,x_data[index],params['sr'],params['n_step'])

            x_data = np.vstack((x_data,pitch_changed))
            y_class = np.hstack((y_class,y_class[index]))
        except:
            pass

    if params['CS_mode']:
        try:
            index = random.sample(range(0,data_size), sampling_size)
            speed_changed = np.apply_along_axis(changing_speed,1,x_data[index],params['rate'])

            x_data = np.vstack((x_data,speed_changed))
            y_class = np.hstack((y_class,y_class[index]))
        except:
            pass


    return x_data,y_class

# Params = {"NJ_mode":True,"noise_factor":0.01,  # parameters for noise injection
#           "ST_mode":True,"shift_length":1000,"shift_direction":"right",  # parameters for shifting time
#           "CP_mode":True,"sr":44,"n_step":0,  # parameters for changing pitch
#           "CS_mode":True,"rate":1.01,  # parameters for changing speed
#           "sampling_rate":0.1  # proportion of sample to apply augmentation
#          }
# # stack the original data and augmentated data
# data_aug,classes_aug = batch_aug(audios,classes,Params)