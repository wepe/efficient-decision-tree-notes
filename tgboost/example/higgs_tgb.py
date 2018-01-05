from tgboost import tgb
import pandas as pd
import numpy as np

train = pd.read_csv('../../data/train.csv').drop(["EventId", "Weight"], axis=1)
val = pd.read_csv('../../data/test.csv').drop(["EventId", "Weight"], axis=1)
train.replace(to_replace=-999., value=np.nan, inplace=True)
train.replace(to_replace='s', value=1, inplace=True)
train.replace(to_replace='b', value=0, inplace=True)
val.replace(to_replace=-999, value=np.nan, inplace=True)
val.replace(to_replace='s', value=1, inplace=True)
val.replace(to_replace='b', value=0, inplace=True)


train_y = train.Label.values
train_X = train.drop('Label', axis=1).values
val_y = val.Label.values
val_X = val.drop('Label', axis=1).values

print train_y.shape, train_y.sum()
print val_y.shape, val_y.sum()

params = {'loss': "logisticloss",
          'eta': 0.3,
          'max_depth': 7,
          'num_boost_round': 500,
          'scale_pos_weight': 1.0,
          'subsample': 0.8,
          'colsample': 0.8,
          'min_sample_split': 50,
          'min_child_weight': 1,
          'reg_lambda': 1,
          'gamma': 0.01,
          'eval_metric': "auc",
          'early_stopping_rounds': 50,
          'maximize': True}

tgb.fit(train_X, train_y, validation_data=(val_X, val_y), **params)