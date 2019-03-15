import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
import argparse

from LASSOBandit import *

def get_data_npy(loadfile='data/train_data.npz'):
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
    # 2. Drop label('Therapeutic Dose of Warfarin'), patient ID column
    df = df.dropna(axis=0, subset=['Therapeutic Dose of Warfarin'])

    # Extract label (warfarin dose) column and discretize it
    thresh1 = 21
    thresh2 = 49
    Y_train = np.array(df['Therapeutic Dose of Warfarin'].values, dtype=int)
    Y_train[Y_train < thresh1] = 0
    Y_train[np.logical_and(Y_train >= thresh1, Y_train <= thresh2)] = 1
    Y_train[Y_train > thresh2] = 2

    uneeded_columns = ['Therapeutic Dose of Warfarin', 'PharmGKB Subject ID']
    df = df.drop(columns=uneeded_columns)

    # convert csv values to nice integer array
    X_train = np.zeros(df.shape, dtype=int)
    col_dtypes = df.dtypes
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

data_file = 'features_old.npz'
if os.path.isfile(data_file): 
    X, Y = get_data_npy(data_file)
else:
    X, Y = get_data_csv(data_file)

shuffle = False
X_shuffled = X
Y_shuffled = Y
num_samples, N = X.shape
if shuffle:
    shuffled = np.random.permutation(np.column_stack((Y, X)).T).T
    Y_shuffled = shuffled[:, 0]
    X_shuffled = shuffled[:, 1:]
print(X.shape)
action_dim = 3

delta = 0.1
alpha = 1 + np.sqrt(np.log(2/delta)/2)
#def __init__(self, K, d, num_samples, q, h, lambda_1, lambda_2_0):
q = 1
h = .25
lambda_1 = .1
lambda_2_0 = .1
agent = LASSOBandit(action_dim, N, q, h, lambda_1, lambda_2_0)

print("Simulating", num_samples, "patients...")

predictions = []
rewards = []
wrong_count = 0
for i in range(num_samples):
    X = X_shuffled[i, :]
    prediction = agent.predict(X) - 1
    if prediction == Y_shuffled[i]:
        reward = 0
    else:
        reward = -1
        wrong_count += 1
    agent.update_reward(reward)
    predictions.append(prediction)
    rewards.append(reward)

#intervals = agent.confidence_intervals

if 1:
    logfile = 'log/log.txt'
    with open(logfile, 'w') as f:
        for i in range(num_samples):
            line = f'reward: {rewards[i]},\t prediction: {predictions[i]},\t true_value: {int(Y_shuffled[i])}\n'
            f.write(line)

performance = 1 - float(wrong_count) / num_samples
print('Performance:', performance)
