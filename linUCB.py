import pandas as pd
import csv

from functools import reduce
import numpy as np


original = pd.read_csv("data/warfarin.csv")
dropna_cols = ['Age', 'Height (cm)', 'Weight (kg)',
          'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)',
          'Rifampin or Rifampicin', 'Amiodarone (Cordarone)']
excluded_from_dropna = ['Race']

df = original[dropna_cols + excluded_from_dropna]
df.dropna(subset=dropna_cols, inplace=True)

df['Therapeutic Dose of Warfarin'] = original['Therapeutic Dose of Warfarin'][df.index] / 7.0

weight_of_race = {
    'White': 0,
    'Black or African American': -.406,
    'Asian': -.6752,
    'Unknown': .0443,
    'nan': .0443
}

df['Age'] = df['Age'].map(lambda a: int(a[0]))

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

Y = df['Therapeutic Dose of Warfarin'].map(
        dosage_bucket).values
df = df.drop(columns=['Race', 'Therapeutic Dose of Warfarin'])

print(df.head(5))

X = df.values
np.savez("features.npz", X_train=X, Y_train=Y)
