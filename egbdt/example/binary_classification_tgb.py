from egbdt import tgb
import pandas as pd

train = pd.read_csv('../../train.csv')
train = train.sample(frac=1.0, axis=0)  # shuffle the data
train.fillna(-999, inplace=True)

val = train.iloc[0:5000]
train = train.iloc[5000:]


train_y = train.label.values
train_X = train.drop('label', axis=1).values
val_y = val.label.values
val_X = val.drop('label', axis=1).values
del train, val

print train_X.shape, val_X.shape

params = {'loss': "logisticloss",
          'eta': 0.3,
          'max_depth': 6,
          'num_boost_round': 500,
          'scale_pos_weight': 1.0,
          'subsample': 0.7,
          'colsample': 0.7,
          'min_sample_split': 10,
          'min_child_weight': 2,
          'reg_lambda': 10,
          'gamma': 0,
          'eval_metric': "error",
          'early_stopping_rounds': 20,
          'maximize': False}

tgb.fit(train_X, train_y, validation_data=(val_X, val_y), **params)

"""
TGBoost round 0, train-error is 0.230885714286, val-error is 0.236, time cost 17.919107914s
TGBoost round 1, train-error is 0.227257142857, val-error is 0.2382, time cost 18.4163079262s
TGBoost round 2, train-error is 0.226771428571, val-error is 0.2372, time cost 17.6617529392s
TGBoost round 3, train-error is 0.227885714286, val-error is 0.2368, time cost 18.1195938587s
TGBoost round 4, train-error is 0.224114285714, val-error is 0.2352, time cost 18.0385129452s
TGBoost round 5, train-error is 0.2242, val-error is 0.2334, time cost 18.870816946s
TGBoost round 6, train-error is 0.224228571429, val-error is 0.235, time cost 17.9746758938s
TGBoost round 7, train-error is 0.225457142857, val-error is 0.2344, time cost 21.3916988373s
TGBoost round 8, train-error is 0.226514285714, val-error is 0.235, time cost 20.8191080093s
TGBoost round 9, train-error is 0.225514285714, val-error is 0.2322, time cost 17.6918649673s
TGBoost round 10, train-error is 0.2254, val-error is 0.2326, time cost 18.6689469814s
TGBoost round 11, train-error is 0.224257142857, val-error is 0.2318, time cost 18.4976451397s
TGBoost round 12, train-error is 0.224485714286, val-error is 0.233, time cost 20.3194348812s
TGBoost round 13, train-error is 0.223714285714, val-error is 0.2302, time cost 18.8383648396s
TGBoost round 14, train-error is 0.223085714286, val-error is 0.232, time cost 20.0607531071s
TGBoost round 15, train-error is 0.223114285714, val-error is 0.233, time cost 19.4844110012s
TGBoost round 16, train-error is 0.223114285714, val-error is 0.2336, time cost 17.4264760017s
TGBoost round 17, train-error is 0.223771428571, val-error is 0.2336, time cost 19.1870779991s
TGBoost round 18, train-error is 0.223971428571, val-error is 0.2328, time cost 17.4278280735s
TGBoost round 19, train-error is 0.223571428571, val-error is 0.2332, time cost 16.7509601116s
TGBoost round 20, train-error is 0.223514285714, val-error is 0.2334, time cost 18.9180390835s
TGBoost round 21, train-error is 0.224285714286, val-error is 0.2348, time cost 19.2357711792s
TGBoost round 22, train-error is 0.222457142857, val-error is 0.2324, time cost 17.1019370556s
TGBoost round 23, train-error is 0.222342857143, val-error is 0.2314, time cost 18.3335280418s
TGBoost round 24, train-error is 0.221714285714, val-error is 0.2294, time cost 19.4953269958s
TGBoost round 25, train-error is 0.221514285714, val-error is 0.2288, time cost 16.5090939999s
TGBoost round 26, train-error is 0.221, val-error is 0.2288, time cost 17.7107200623s
TGBoost round 27, train-error is 0.2202, val-error is 0.2274, time cost 16.563462019s
TGBoost round 28, train-error is 0.2198, val-error is 0.2274, time cost 19.8756830692s
TGBoost round 29, train-error is 0.219285714286, val-error is 0.2264, time cost 20.5743000507s
TGBoost round 30, train-error is 0.219228571429, val-error is 0.2258, time cost 19.7626709938s
TGBoost round 31, train-error is 0.218771428571, val-error is 0.2258, time cost 20.9459028244s
TGBoost round 32, train-error is 0.218257142857, val-error is 0.2246, time cost 17.5552449226s
TGBoost round 33, train-error is 0.216542857143, val-error is 0.2224, time cost 19.3163299561s
TGBoost round 34, train-error is 0.216371428571, val-error is 0.2224, time cost 18.9449779987s
TGBoost round 35, train-error is 0.214885714286, val-error is 0.2188, time cost 17.3829491138s
TGBoost round 36, train-error is 0.214371428571, val-error is 0.218, time cost 17.608743906s
TGBoost round 37, train-error is 0.214285714286, val-error is 0.217, time cost 18.7361879349s
TGBoost round 38, train-error is 0.214485714286, val-error is 0.2166, time cost 19.1980118752s
TGBoost round 39, train-error is 0.214257142857, val-error is 0.2166, time cost 18.1930348873s
TGBoost round 40, train-error is 0.214228571429, val-error is 0.2166, time cost 18.262691021s
TGBoost round 41, train-error is 0.2142, val-error is 0.2196, time cost 15.4086480141s
TGBoost round 42, train-error is 0.214542857143, val-error is 0.2194, time cost 18.873953104s
TGBoost round 43, train-error is 0.214514285714, val-error is 0.2194, time cost 17.7430160046s
TGBoost round 44, train-error is 0.214371428571, val-error is 0.2184, time cost 20.1922318935s
TGBoost round 45, train-error is 0.213742857143, val-error is 0.2194, time cost 15.6589348316s
TGBoost round 46, train-error is 0.213628571429, val-error is 0.2194, time cost 19.1324648857s
TGBoost round 47, train-error is 0.213914285714, val-error is 0.219, time cost 21.0888469219s
TGBoost round 48, train-error is 0.213714285714, val-error is 0.2198, time cost 18.7675480843s
TGBoost round 49, train-error is 0.213914285714, val-error is 0.22, time cost 22.0127091408s
TGBoost round 50, train-error is 0.2138, val-error is 0.2202, time cost 21.4829149246s
TGBoost round 51, train-error is 0.212971428571, val-error is 0.218, time cost 14.8317320347s
TGBoost round 52, train-error is 0.213057142857, val-error is 0.2176, time cost 18.9493699074s
TGBoost round 53, train-error is 0.213914285714, val-error is 0.2204, time cost 18.8809850216s
TGBoost round 54, train-error is 0.213914285714, val-error is 0.2202, time cost 21.2961080074s
TGBoost round 55, train-error is 0.213028571429, val-error is 0.2182, time cost 15.7712600231s
TGBoost round 56, train-error is 0.212857142857, val-error is 0.2192, time cost 19.0951440334s
TGBoost round 57, train-error is 0.212714285714, val-error is 0.2194, time cost 17.8546860218s
TGBoost round 58, train-error is 0.2126, val-error is 0.2198, time cost 17.8603348732s
TGBoost round 59, train-error is 0.212342857143, val-error is 0.2192, time cost 19.5755591393s
TGBoost training Stop, best round is 38, best val-error is 0.2166s


"""