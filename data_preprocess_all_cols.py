import re
import pandas as pd
import csv

from functools import reduce
import numpy as np


original = pd.read_csv("data/warfarin.csv")
df = original
df.dropna(axis=0, subset=['Therapeutic Dose of Warfarin'], inplace=True)
num_rows = len(df)

enzyme_status_cols = ['Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)',
                      'Rifampin or Rifampicin']

df['Enzyme Status'] = df[enzyme_status_cols].apply(
    lambda x: int(reduce(np.logical_xor, x)), axis=1)
    
drop_cols = [
        'PharmGKB Subject ID',
        'Subject Reached Stable Dose of Warfarin',
        'INR on Reported Therapeutic Dose of Warfarin',
        'Comorbidities',
        'Medications',
        'Indication for Warfarin Treatment',
        'Estimated Target INR Range Based on Indication',
        ]

drop_thresh = .5
for col in df.columns:
    if df[col].isnull().sum() > num_rows * drop_thresh:
        drop_cols.append(col)

df.drop(columns=drop_cols, inplace=True)

numerical_cols = [
        'Age',
        'Height (cm)',
        'Weight (kg)',
        ]
df.loc[:, numerical_cols].fillna(0, inplace=True)

def map_age(a):
    try: 
        return int(a[0]) 
    except: 
        return a
 
df['Age'] = df['Age'].map(map_age)

# df = df.applymap(lambda s:s.lower() if type(s) == str else s)
# multi_categorical_cols = [
#         'Indication for Warfarin Treatment',
#         ]

# def to_comma_separated(s, delims):
#     return ','.join(re.sub('['+''.join(delims)+']( )+', '_', str(s)).split('_'))


# for col in multi_categorical_cols:
#     df[col] = df[col].map(
#             lambda x: to_comma_separated(x, [',', ';']))
#     df = df.join(df[col].str.get_dummies(sep=',').add_prefix(col+'_'))
#     df.drop(columns=[col], inplace=True)

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
        'VKORC1 3730 consensus',
        'Gender',
        'Diabetes',
        'Congestive Heart Failure and/or Cardiomyopathy',
        'Valve Replacement',
        'Aspirin',
        'Simvastatin (Zocor)',
        'Amiodarone (Cordarone)',
        'Current Smoker',

        ]
        # 'VKORC1 2255 consensus',
        # 'VKORC1 497 consensus',
        # 'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
        # 'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
df = pd.get_dummies(df, dummy_na=True, columns=categorical_cols)

df['Therapeutic Dose of Warfarin'] = df['Therapeutic Dose of Warfarin'] / 7.0
df['bias'] = 1.0

def dosage_bucket(daily_dosage):
    if daily_dosage < 3:
        return 0
    elif daily_dosage <= 7:
        return 1
    else:
        return 2

Y = df['Therapeutic Dose of Warfarin'].map(dosage_bucket).values
df = df.drop(columns=['Therapeutic Dose of Warfarin'])
df = df.fillna(0)

s1f_cols = [
    'Age',
    'Height (cm)',
    'Weight (kg)',
    'Enzyme Status',
    'Amiodarone (Cordarone)_1.0',
    'Race_White',
    'Race_Black or African American',
    'Race_Asian',
    'Race_Unknown',
    'Race_nan',
    ]
    
normalize_features = df[['Age', 'Height (cm)', 'Weight (kg)']]
df[['Age', 'Height (cm)', 'Weight (kg)']] = (normalize_features-normalize_features.mean())/normalize_features.std()

cols = df.columns
other_cols = [x for x in cols if x not in s1f_cols]
df=df.reindex(columns=s1f_cols + other_cols)

X = df.values
print(df.sample(5))
print(X.shape)
np.savez("features.npz", X_train=X, Y_train=Y)
