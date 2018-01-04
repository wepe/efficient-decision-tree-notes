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
          'max_depth': 5,
          'num_boost_round': 500,
          'scale_pos_weight': 1.0,
          'subsample': 0.8,
          'colsample': 0.8,
          'min_sample_split': 50,
          'min_child_weight': 1,
          'reg_lambda': 10,
          'gamma': 0.01,
          'eval_metric': "error",
          'early_stopping_rounds': 20,
          'maximize': False}

tgb.fit(train_X, train_y, validation_data=(val_X, val_y), **params)

"""
TGBoost round 0, train-error is 0.238742857143, val-error is 0.2456, time cost 3.71701383591s
TGBoost round 1, train-error is 0.230285714286, val-error is 0.2338, time cost 3.69342088699s
TGBoost round 2, train-error is 0.2306, val-error is 0.2344, time cost 3.70841407776s
TGBoost round 3, train-error is 0.231542857143, val-error is 0.231, time cost 3.70805692673s
TGBoost round 4, train-error is 0.230828571429, val-error is 0.2314, time cost 3.72947621346s
TGBoost round 5, train-error is 0.229085714286, val-error is 0.2276, time cost 3.75900292397s
TGBoost round 6, train-error is 0.23, val-error is 0.2302, time cost 3.70230102539s
TGBoost round 7, train-error is 0.2288, val-error is 0.2292, time cost 3.73828411102s
TGBoost round 8, train-error is 0.228057142857, val-error is 0.229, time cost 3.75026297569s
TGBoost round 9, train-error is 0.2266, val-error is 0.2296, time cost 3.70752620697s
TGBoost round 10, train-error is 0.227, val-error is 0.2318, time cost 3.68288016319s
TGBoost round 11, train-error is 0.227657142857, val-error is 0.2322, time cost 3.71141815186s
TGBoost round 12, train-error is 0.2264, val-error is 0.231, time cost 3.69189381599s
TGBoost round 13, train-error is 0.225828571429, val-error is 0.2292, time cost 3.81694102287s
TGBoost round 14, train-error is 0.225628571429, val-error is 0.2292, time cost 3.71179795265s
TGBoost round 15, train-error is 0.2256, val-error is 0.2284, time cost 3.72830486298s
TGBoost round 16, train-error is 0.224228571429, val-error is 0.2276, time cost 3.77151203156s
TGBoost round 17, train-error is 0.224057142857, val-error is 0.2258, time cost 3.82595896721s
TGBoost round 18, train-error is 0.2234, val-error is 0.226, time cost 3.70994997025s
TGBoost round 19, train-error is 0.223, val-error is 0.2252, time cost 3.75701904297s
TGBoost round 20, train-error is 0.2206, val-error is 0.2218, time cost 3.68910098076s
TGBoost round 21, train-error is 0.220914285714, val-error is 0.223, time cost 3.7491979599s
TGBoost round 22, train-error is 0.217942857143, val-error is 0.2202, time cost 3.78888297081s
TGBoost round 23, train-error is 0.218028571429, val-error is 0.221, time cost 3.71157193184s
TGBoost round 24, train-error is 0.218285714286, val-error is 0.2198, time cost 3.73786687851s
TGBoost round 25, train-error is 0.218428571429, val-error is 0.2212, time cost 3.77874898911s
TGBoost round 26, train-error is 0.218257142857, val-error is 0.2198, time cost 3.73246216774s
TGBoost round 27, train-error is 0.217971428571, val-error is 0.2188, time cost 3.5935549736s
TGBoost round 28, train-error is 0.217257142857, val-error is 0.2182, time cost 3.63868999481s
TGBoost round 29, train-error is 0.2168, val-error is 0.2178, time cost 3.72400403023s
TGBoost round 30, train-error is 0.2164, val-error is 0.2172, time cost 3.71992492676s
TGBoost round 31, train-error is 0.216142857143, val-error is 0.217, time cost 3.69051504135s
TGBoost round 32, train-error is 0.216628571429, val-error is 0.217, time cost 3.7802131176s
TGBoost round 33, train-error is 0.2162, val-error is 0.2164, time cost 3.67879986763s
TGBoost round 34, train-error is 0.216285714286, val-error is 0.217, time cost 3.69256806374s
TGBoost round 35, train-error is 0.216114285714, val-error is 0.2166, time cost 3.66421985626s
TGBoost round 36, train-error is 0.215771428571, val-error is 0.2158, time cost 3.74409604073s
TGBoost round 37, train-error is 0.214542857143, val-error is 0.2164, time cost 3.69231891632s
TGBoost round 38, train-error is 0.214857142857, val-error is 0.2166, time cost 3.70529198647s
TGBoost round 39, train-error is 0.214257142857, val-error is 0.2138, time cost 3.76646089554s
TGBoost round 40, train-error is 0.213885714286, val-error is 0.2136, time cost 3.74382400513s
TGBoost round 41, train-error is 0.213542857143, val-error is 0.213, time cost 3.63991785049s
TGBoost round 42, train-error is 0.213428571429, val-error is 0.215, time cost 3.77041006088s
TGBoost round 43, train-error is 0.212685714286, val-error is 0.215, time cost 3.62445902824s
TGBoost round 44, train-error is 0.212685714286, val-error is 0.2144, time cost 3.71449494362s
TGBoost round 45, train-error is 0.212714285714, val-error is 0.2144, time cost 3.66005396843s
TGBoost round 46, train-error is 0.212371428571, val-error is 0.2144, time cost 3.66620707512s
TGBoost round 47, train-error is 0.212542857143, val-error is 0.2138, time cost 3.67622303963s
TGBoost round 48, train-error is 0.212142857143, val-error is 0.214, time cost 3.70943689346s
TGBoost round 49, train-error is 0.2122, val-error is 0.2136, time cost 3.61202192307s
TGBoost round 50, train-error is 0.211942857143, val-error is 0.2134, time cost 3.66828107834s
TGBoost round 51, train-error is 0.211714285714, val-error is 0.2136, time cost 3.65024113655s
TGBoost round 52, train-error is 0.211342857143, val-error is 0.2122, time cost 3.66291499138s
TGBoost round 53, train-error is 0.211114285714, val-error is 0.2124, time cost 3.73679614067s
TGBoost round 54, train-error is 0.211428571429, val-error is 0.2132, time cost 3.75595188141s
TGBoost round 55, train-error is 0.2114, val-error is 0.214, time cost 3.66404008865s
TGBoost round 56, train-error is 0.2112, val-error is 0.2136, time cost 3.78315210342s
TGBoost round 57, train-error is 0.210628571429, val-error is 0.2144, time cost 3.60629796982s
TGBoost round 58, train-error is 0.210771428571, val-error is 0.2138, time cost 3.68571186066s
TGBoost round 59, train-error is 0.210828571429, val-error is 0.2144, time cost 3.72975087166s
TGBoost round 60, train-error is 0.210657142857, val-error is 0.2142, time cost 3.66297101974s
TGBoost round 61, train-error is 0.210742857143, val-error is 0.2144, time cost 3.63383293152s
TGBoost round 62, train-error is 0.2104, val-error is 0.2146, time cost 3.67448306084s
TGBoost round 63, train-error is 0.2102, val-error is 0.2146, time cost 3.60910105705s
TGBoost round 64, train-error is 0.210228571429, val-error is 0.2142, time cost 3.67948198318s
TGBoost round 65, train-error is 0.210257142857, val-error is 0.2142, time cost 3.70058321953s
TGBoost round 66, train-error is 0.210085714286, val-error is 0.2146, time cost 3.71190094948s
TGBoost round 67, train-error is 0.210257142857, val-error is 0.2146, time cost 3.60603618622s
TGBoost round 68, train-error is 0.210257142857, val-error is 0.2146, time cost 3.68661594391s
TGBoost round 69, train-error is 0.2102, val-error is 0.2148, time cost 3.65223407745s
TGBoost round 70, train-error is 0.209657142857, val-error is 0.2146, time cost 3.61065888405s
TGBoost round 71, train-error is 0.209142857143, val-error is 0.2138, time cost 3.60764408112s
TGBoost round 72, train-error is 0.209342857143, val-error is 0.2132, time cost 3.63329005241s
TGBoost round 73, train-error is 0.2092, val-error is 0.2138, time cost 3.67511677742s
TGBoost round 74, train-error is 0.209142857143, val-error is 0.2136, time cost 3.65356612206s
TGBoost round 75, train-error is 0.209028571429, val-error is 0.2136, time cost 3.68754291534s
TGBoost round 76, train-error is 0.209, val-error is 0.2138, time cost 3.61171102524s
TGBoost round 77, train-error is 0.208857142857, val-error is 0.2138, time cost 3.67216515541s
TGBoost round 78, train-error is 0.208828571429, val-error is 0.2138, time cost 3.6556289196s
TGBoost round 79, train-error is 0.208914285714, val-error is 0.2128, time cost 3.60161304474s
TGBoost round 80, train-error is 0.209028571429, val-error is 0.2128, time cost 3.76077389717s
TGBoost round 81, train-error is 0.208857142857, val-error is 0.2128, time cost 3.67292404175s
TGBoost round 82, train-error is 0.208857142857, val-error is 0.2128, time cost 3.57409310341s
TGBoost round 83, train-error is 0.209028571429, val-error is 0.2124, time cost 3.7161090374s
TGBoost round 84, train-error is 0.209114285714, val-error is 0.2128, time cost 3.63063502312s
TGBoost round 85, train-error is 0.208685714286, val-error is 0.2128, time cost 3.67365813255s
TGBoost round 86, train-error is 0.208685714286, val-error is 0.213, time cost 3.58945894241s
TGBoost round 87, train-error is 0.208714285714, val-error is 0.2128, time cost 3.59758901596s
TGBoost round 88, train-error is 0.209028571429, val-error is 0.213, time cost 3.61627411842s
TGBoost round 89, train-error is 0.208742857143, val-error is 0.2134, time cost 3.62249994278s
TGBoost round 90, train-error is 0.208542857143, val-error is 0.2134, time cost 3.56504416466s
TGBoost round 91, train-error is 0.208485714286, val-error is 0.213, time cost 3.7109978199s
TGBoost round 92, train-error is 0.208457142857, val-error is 0.213, time cost 3.65219807625s
TGBoost round 93, train-error is 0.208371428571, val-error is 0.2128, time cost 3.6589820385s
TGBoost round 94, train-error is 0.208228571429, val-error is 0.2132, time cost 3.67305898666s
TGBoost round 95, train-error is 0.2082, val-error is 0.2132, time cost 3.66815090179s
TGBoost round 96, train-error is 0.2082, val-error is 0.2132, time cost 3.73147678375s
TGBoost round 97, train-error is 0.208171428571, val-error is 0.2132, time cost 3.63195705414s
TGBoost round 98, train-error is 0.208057142857, val-error is 0.2132, time cost 3.67149996758s
TGBoost round 99, train-error is 0.208285714286, val-error is 0.2132, time cost 3.66407799721s
TGBoost round 100, train-error is 0.208228571429, val-error is 0.2128, time cost 3.67459392548s
TGBoost round 101, train-error is 0.208171428571, val-error is 0.2128, time cost 3.6504199504
"""