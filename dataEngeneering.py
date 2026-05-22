import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('Emergencias_UNGRD_Translated.xlsx')

df.columns = df.columns.str.upper().str.strip()

df = df.loc[:, ~df.columns.str.contains('^UNNAMED')]

df.drop_duplicates(inplace=True)

drop_cols = [
    'RENT SUBSIDY',
    'NON-FOOD ASSISTANCE',
    'FOOD SUPPORT',
    'CONSTRUCTION MATERIALS',
    'BIG BAGS',
    'EMERGENCY WORKS',
    'WATER TANKERS - PUMPS - WATER TREATMENT PLANT',
    'MACHINE HOURS',
    'OTHER-AFFECTATION',
    'MISSING',
    'HEALTH CENTERS',
    'SEWER SYSTEM',
    'COMMUNITY CENTERS',
    'PEDESTRIAN BRIDGES',
    'AQUEDUCT',
    'VEHICLE BRIDGES'
]

df.drop(columns=drop_cols, inplace=True, errors='ignore')

text_cols = df.select_dtypes(include=['object', 'string']).columns

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

possible_numeric_cols = [
    'DECEASED',
    'INJURED',
    'PEOPLE',
    'FAMILIES',
    'DESTROYED HOUSES',
    'DAMAGED HOUSES',
    'HECTARES'
]

for col in possible_numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df[possible_numeric_cols] = df[possible_numeric_cols].fillna(0)

df[text_cols] = df[text_cols].fillna('Unknown')

df['DATE'] = pd.to_datetime(
    df['DATE'],
    errors='coerce',
    dayfirst=True
)

df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month

df['TOTAL_AFFECTED'] = (
    df['PEOPLE'] +
    df['FAMILIES']
)

df['TOTAL_HOUSING_DAMAGE'] = (
    df['DESTROYED HOUSES'] +
    df['DAMAGED HOUSES']
)

df['SEVERITY_INDEX'] = (
    (df['DECEASED'] * 5) +
    (df['INJURED'] * 3) +
    df['TOTAL_AFFECTED'] +
    (df['TOTAL_HOUSING_DAMAGE'] * 2)
)

df['MUNICIPALITY_EVENT_COUNT'] = (
    df.groupby('MUNICIPALITY')['EVENT']
    .transform('count')
)

df['EVENT_DIVERSITY'] = (
    df.groupby('MUNICIPALITY')['EVENT']
    .transform('nunique')
)

df['AVG_SEVERITY_MUNICIPALITY'] = (
    df.groupby('MUNICIPALITY')['SEVERITY_INDEX']
    .transform('mean')
)

df['RAINY_SEASON'] = (
    df['MONTH']
    .isin([4, 5, 10, 11])
    .astype(int)
)

df['VULNERABILITY_INDEX'] = (
    (df['MUNICIPALITY_EVENT_COUNT'] * 0.4) +
    (df['AVG_SEVERITY_MUNICIPALITY'] * 0.4) +
    (df['EVENT_DIVERSITY'] * 0.2)
)

p33 = df['SEVERITY_INDEX'].quantile(0.33)
p66 = df['SEVERITY_INDEX'].quantile(0.66)

df['VULNERABILITY_LEVEL'] = pd.cut(
    df['SEVERITY_INDEX'],
    bins=[-np.inf, p33, p66, np.inf],
    labels=['Low', 'Medium', 'High']
)


encoder = LabelEncoder()

for col in ['DEPARTMENT', 'MUNICIPALITY', 'EVENT']:
    df[col] = encoder.fit_transform(df[col])

scale_cols = [
    'TOTAL_AFFECTED',
    'TOTAL_HOUSING_DAMAGE',
    'SEVERITY_INDEX',
    'MUNICIPALITY_EVENT_COUNT',
    'EVENT_DIVERSITY',
    'AVG_SEVERITY_MUNICIPALITY',
    'VULNERABILITY_INDEX'
]

df_scaled = df.copy()

scaler = StandardScaler()

df_scaled[scale_cols] = scaler.fit_transform(
    df_scaled[scale_cols]
)

df = df.round(2)

df_scaled = df_scaled.round(2)
zero_percentage = (df == 0).mean() * 100
print(
    zero_percentage.sort_values(ascending=False)
)
df.to_excel('UNGRD_Cleaned.xlsx', index=False)
df_scaled.to_excel('UNGRD_Scaled.xlsx', index=False)