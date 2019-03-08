import pandas as pd
import numpy as np
import os.path

from linUCBHybridAgent import *

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

data_file = 'features.npz'
if os.path.isfile(data_file): 
    X, Y = get_data_npy(data_file)
else:
    X, Y = get_data_csv(data_file)

num_samples, N = X.shape
shuffled = np.random.permutation(np.column_stack((Y, X)))
Y_shuffled = shuffled[:, 0]
X_shuffled = shuffled[:, 1:]

action_dim = 3

delta = 0.1
alpha = 1 + np.sqrt(np.log(2/delta)/2)
agent = LinUCBHybridAgent(alpha, action_dim, N, N)

print("Simulating", num_samples, "patients...")

predictions = []
rewards = []
for i in range(num_samples):
    X = np.expand_dims(X_shuffled[i, :], axis=1)
    prediction = agent.predict(X, X)
    reward = 0 if prediction == Y_shuffled[i] else -1
    agent.update_reward(reward, prediction, X, X)
    predictions.append(prediction)
    rewards.append(reward)

#intervals = agent.confidence_intervals

if 1:
    logfile = 'log/log.txt'
    with open(logfile, 'w') as f:
        for i in range(num_samples):
            line = f'reward: {rewards[i]},\t prediction: {predictions[i]}\n'
            f.write(line)

performance = float(np.sum(rewards) + num_samples) / num_samples
print('Performance:', performance)
