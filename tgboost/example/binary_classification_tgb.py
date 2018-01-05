from tgboost import tgb
import pandas as pd

train = pd.read_csv('../../train.csv')
val = pd.read_csv('../../test.csv')

train_y = train.label.values
train_X = train.drop('label', axis=1).values
val_y = val.label.values
val_X = val.drop('label', axis=1).values

print train_X.shape, val_X.shape

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
          'gamma': 0.1,
          'eval_metric': "error",
          'early_stopping_rounds': 100,
          'maximize': False}

tgb.fit(train_X, train_y, validation_data=(val_X, val_y), **params)

