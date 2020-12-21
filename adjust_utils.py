import math
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau


def power_decay_1(epoch):
    lr_base = 0.001
    drop = 0.4
    epochs_drop = 15.0

    # 0.001 * 0.4^((1 + epoch) / 15)
    lr = lr_base * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    if (lr < 4e-5):
        lr = 4e-5

    print('Changing learning rate to {}'.format(lr))
    return lr

def power_decay_2(epoch):
    lr_base = 0.001
    lr_power = 0.9
    epochs = 60

    # 0.001 * (1 - epoch / 15)^0.9
    lr = lr_base * math.pow((1 - float(epoch) / epochs), lr_power)
    if (lr < 4e-5):
        lr = 4e-5

    print('Changing learning rate to {}'.format(lr))
    return lr

def exp_decay(epoch):
    lr_base = 0.001
    lr_power = 0.9
    # (0.001^0.9)^(epoch + 1)
    lr = math.pow(math.pow(lr_base, lr_power), (epoch + 1))
    if (lr < 4e-5):
        lr = 4e-5

    print('Changing learning rate to {}'.format(lr))
    return lr

def multi_decay(decay_frac=0.1):
    lr = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=decay_frac, patience=3, min_lr=4e-5)
    return lr
