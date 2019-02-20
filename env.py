import pandas as pd
import numpy as np


def get_data_npy(loadfile=None):
    # load existing numpy data
    data = np.load(loadfile)
    X_train = data['X_train']
    Y_train = data['Y_train']
    data.close()
    return X_train, Y_train


def get_data_csv(savefile=None):
    # reading csv file
    filename = "data/warfarin.csv"
    df = pd.read_csv(filename)

    # Preprocess the data
    # 1. Drop patients with no warfarin rec
    # 2. Extract label (warfarin dose) column
    # 3. Drop patient ID column
    df = df.dropna(axis=0, subset=['Therapeutic Dose of Warfarin'])

    # TODO: Bucket y values into low, med, high
    Y_train = df['Therapeutic Dose of Warfarin'].values
    df = df.drop(columns='Therapeutic Dose of Warfarin')

    uneeded_columns = ['PharmGKB Subject ID']
    df = df.drop(columns=uneeded_columns)

    # convert csv values to nice integer array
    X_train = np.zeros(df.shape, dtype=int)
    col = 0
    for name, values in df.iteritems():
        values_to_ints = df[name].unique()

        next_int = 0
        for row, val in enumerate(values):
            X_train[row, col] = np.where(values_to_ints == val)[0][0] if val in values_to_ints else -1
        col += 1

    # save preprocessed data to file
    if savefile:
        np.savez(savefile, X_train=X_train, Y_train=Y_train)

    return X_train, Y_train
