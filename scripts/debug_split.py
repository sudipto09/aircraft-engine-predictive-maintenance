import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load data
cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + \
       [f'sensor_{i}' for i in range(1, 22)]

df = pd.read_csv(
    'data/raw/train_FD001.txt',
    delim_whitespace=True,
    header=None,
    names=cols
)

# Create RUL
max_cycles = df.groupby('engine_id')['cycle'].max()
df['RUL'] = df['engine_id'].map(max_cycles) - df['cycle']

# Engine-wise split
engine_ids = df['engine_id'].unique()
train_engines, val_engines = train_test_split(
    engine_ids, test_size=0.2, random_state=42
)

train_df = df[df['engine_id'].isin(train_engines)]
val_df   = df[df['engine_id'].isin(val_engines)]

print("Train engines:", train_df['engine_id'].nunique())
print("Val engines:", val_df['engine_id'].nunique())
print("Engine overlap:", set(train_engines) & set(val_engines))

# Select sensors
SENSORS = [
    'sensor_2','sensor_3','sensor_4',
    'sensor_7','sensor_8','sensor_9',
    'sensor_11','sensor_12','sensor_13',
    'sensor_15','sensor_17','sensor_20','sensor_21'
]

X_train = train_df[SENSORS]
y_train = train_df['RUL']

X_val = val_df[SENSORS]
y_val = val_df['RUL']

print("Train rows:", X_train.shape[0])
print("Val rows:", X_val.shape[0])

# Impute
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)

# Dummy model
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Dummy RMSE:", rmse)
