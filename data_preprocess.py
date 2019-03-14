import pandas as pd
import csv

from functools import reduce
import numpy as np


original = pd.read_csv("data/warfarin.csv")
dropna_cols = ['Age', 'Height (cm)', 'Weight (kg)',
          'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)',
          'Rifampin or Rifampicin', 'Amiodarone (Cordarone)', 'Therapeutic Dose of Warfarin']
excluded_from_dropna = ['Race']

df = original.dropna(axis=0, subset=['Therapeutic Dose of Warfarin'])
df = df[dropna_cols + excluded_from_dropna]
# df.dropna(subset=dropna_cols, inplace=True)
df['Therapeutic Dose of Warfarin'] = df['Therapeutic Dose of Warfarin'] / 7.0

weight_of_race = {
    'White': 0,
    'Black or African American': -.406,
    'Asian': -.6752,
    'Unknown': .0443,
    'nan': .0443
}

def map_age(a):
    try: 
        return int(a[0]) 
    except: 
        return a

df['Age'] = df['Age'].map(map_age)

for race in weight_of_race:
    df[race] = 0

for index, row in df.iterrows():
    df[df['Race'][index]][index] = 1

def dosage_bucket(daily_dosage):
    if daily_dosage < 3:
        return 0
    elif daily_dosage <= 7:
        return 1
    else:
        return 2

Y = df['Therapeutic Dose of Warfarin'].map(dosage_bucket).values
df = df.drop(columns=['Race', 'Therapeutic Dose of Warfarin'])

#TODO: normalize features to mean 0 variance 1.
normalize_features = df[['Age', 'Height (cm)', 'Weight (kg)']]
df[['Age', 'Height (cm)', 'Weight (kg)']] = (normalize_features-normalize_features.mean())/normalize_features.std()

df = df.fillna(0)

X = df.values
np.savez("features.npz", X_train=X, Y_train=Y)

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