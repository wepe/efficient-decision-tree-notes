import tgboost as tgb

file_training = "/home/wepon/PycharmProjects/data/train_.csv"
file_validation = "/home/wepon/PycharmProjects/data/test_.csv"
file_testing = "/home/wepon/PycharmProjects/data/test_.csv"
early_stopping_rounds = 10
maximize=True
eval_metric="auc"
loss="logloss"
eta=0.3
num_boost_round=1000
max_depth=6
scale_pos_weight=1
subsample=0.8
colsample=0.8
min_child_weight=1
min_sample_split=10
reg_lambda=1.0
gamma=0
num_thread=-1

tgb.run(file_training, file_validation, file_testing, early_stopping_rounds, maximize, eval_metric,
        loss, eta, num_boost_round, max_depth, scale_pos_weight, subsample, colsample, min_child_weight,
        min_sample_split, reg_lambda, gamma, num_thread)