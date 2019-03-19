import pandas as pd
import csv

from functools import reduce
from numpy import logical_xor


original = pd.read_csv("data/warfarin.csv")
dropna_cols = ['Age', 'Height (cm)', 'Weight (kg)',
          'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)',
          'Rifampin or Rifampicin', 'Amiodarone (Cordarone)']
excluded_from_dropna = ['Race']

df = original[dropna_cols + excluded_from_dropna]
df.dropna(subset=dropna_cols, inplace=True)

weight_of_race = {
    'White': 0,
    'Black or African American': -.406,
    'Asian': -.6752,
    'Unknown': .0443,
    'nan': .0443
}

df['Age'] = df['Age'].map(lambda a: int(a[0]))
df['Race'] = df['Race'].map(lambda r: weight_of_race[r])

enzyme_status_cols = ['Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)',
                      'Rifampin or Rifampicin']

df['Enzyme Status'] = df[enzyme_status_cols].apply(
    lambda x: int(reduce(logical_xor, x)), axis=1)
df.drop(columns=enzyme_status_cols, inplace=True)

weight_of_col = {
    'Age': -.2546,
    'Height (cm)': .0118,
    'Weight (kg)': .0134,
    'Race': 1,
    'Enzyme Status': 1.2799,
    'Amiodarone (Cordarone)': -.5695
}

df['Daily Dose'] = 4.0376
for col in df.columns:
    if col != 'Daily Dose':
        df['Daily Dose'] += weight_of_col[col] * df[col]
df['Daily Dose'] = df['Daily Dose']**2 / 7.0
df['Therapeutic Dose of Warfarin'] = original['Therapeutic Dose of Warfarin'][df.index] / 7.0

def dosage_bucket(daily_dosage):
    if daily_dosage < 3:
        return 0
    elif daily_dosage <= 7:
        return 1
    else:
        return 2

df['True Bucket'] = df['Therapeutic Dose of Warfarin'].map(
        dosage_bucket)
df['Predicted Bucket'] = df['Daily Dose'].map(
        dosage_bucket)

#print(df.head(5))
print(len(df))
print("Basline Accuracy: %f" % (sum(df['True Bucket'] == df['Predicted Bucket'])/len(df)))
# Basline Accuracy: 0.650801