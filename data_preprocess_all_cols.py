import re
import pandas as pd
import csv

from functools import reduce
import numpy as np


original = pd.read_csv("data/warfarin.csv")
df = original

num_rows = len(df)

drop_cols = [
        'PharmGKB Subject ID',
        'Subject Reached Stable Dose of Warfarin',
        'INR on Reported Therapeutic Dose of Warfarin',
        'Comorbidities',
        'Medications'
        ]

for col in df.columns:
    if df[col].isnull().sum() > num_rows / 2.:
        drop_cols.append(col)

df.drop(columns=drop_cols, inplace=True)
#for col in df.columns:
#    print(col, df[col].unique())

numerical_cols = [
        'Age',
        'Height (cm)',
        'Weight (kg)',
        'Estimated Target INR Range Based on Indication'
        ]
df.dropna(subset=numerical_cols, inplace=True)

def map_age(a):
    try: 
        return int(a[0]) 
    except: 
        return a
 
df['Age'] = df['Age'].map(map_age)

binary_cols = [
        'Gender',
        'Diabetes',
        'Congestive Heart Failure and/or Cardiomyopathy',
        'Valve Replacement',
        'Aspirin',
        'Simvastatin (Zocor)',
        'Amiodarone (Cordarone)',
        'Current Smoker',
        ]

for col in binary_cols:
    mode = df[col].mode()
    df[col].fillna(df[col].mode()[0], inplace=True)

multi_categorical_cols = [
        'Indication for Warfarin Treatment',
        ]

def to_comma_separated(s, delims):
    return ','.join(re.sub('['+''.join(delims)+']( )+', '_', str(s)).split('_'))


for col in multi_categorical_cols:
    df[col] = df[col].map(
            lambda x: to_comma_separated(x, [',', ';']))
    df = df.join(df[col].str.get_dummies(sep=',').add_prefix(col+'_'))
    df.drop(columns=[col], inplace=True)

categorical_cols = [
        'Race',
        'Ethnicity',
        'Cyp2C9 genotypes',
        'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
        'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
        'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
        'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
        'CYP2C9 consensus',
        'VKORC1 -1639 consensus',
        'VKORC1 1173 consensus',
        'VKORC1 1542 consensus',
        'VKORC1 3730 consensus'
        ]
df = pd.get_dummies(df, dummy_na=True, columns=categorical_cols)





#dropna_cols = ['Age', 'Height (cm)', 'Weight (kg)',
#          'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)',
#          'Rifampin or Rifampicin', 'Amiodarone (Cordarone)', 'Therapeutic Dose of Warfarin']
#excluded_from_dropna = ['Race']
#
df['Therapeutic Dose of Warfarin'] = df['Therapeutic Dose of Warfarin'] / 7.0
#
#weight_of_race = {
#    'White': 0,
#    'Black or African American': -.406,
#    'Asian': -.6752,
#    'Unknown': .0443,
#    'nan': .0443
#}
#
#
#for race in weight_of_race:
#    df[race] = 0
#
#for index, row in df.iterrows():
#    df[df['Race'][index]][index] = 1
#
def dosage_bucket(daily_dosage):
    if daily_dosage < 3:
        return 0
    elif daily_dosage <= 7:
        return 1
    else:
        return 2

Y = df['Therapeutic Dose of Warfarin'].map(dosage_bucket).values
#df = df.drop(columns=['Race', 'Therapeutic Dose of Warfarin'])
#
##TODO: normalize features to mean 0 variance 1.
normalize_features = df[['Age', 'Height (cm)', 'Weight (kg)']]
df[['Age', 'Height (cm)', 'Weight (kg)']] = (normalize_features-normalize_features.mean())/normalize_features.std()
#
#df = df.fillna(0)
#
X = df.values
#np.savez("features.npz", X_train=X, Y_train=Y)
