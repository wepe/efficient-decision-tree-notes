from tgboost import tgb
import pandas as pd

train = pd.read_csv('../../train.csv')
train = train.sample(frac=1.0, axis=0)  # shuffle the data

val = train.iloc[0:5000]
train = train.iloc[5000:]


train_y = train.label.values
train_X = train.drop('label', axis=1).values
val_y = val.label.values
val_X = val.drop('label', axis=1).values

print train_X.shape, val_X.shape

params = {'loss': "logisticloss",
          'eta': 0.3,
          'max_depth': 8,
          'num_boost_round': 500,
          'scale_pos_weight': 1.0,
          'subsample': 0.8,
          'colsample': 0.8,
          'min_sample_split': 30,
          'min_child_weight': 1,
          'reg_lambda': 1,
          'gamma': 0.01,
          'eval_metric': "error",
          'early_stopping_rounds': 50,
          'maximize': False}

tgb.fit(train_X, train_y, validation_data=(val_X, val_y), **params)

