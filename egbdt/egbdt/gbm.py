import numpy as np
from loss import SquareLoss, LogisticLoss, CustomizeLoss
from tree import Tree
from metric import get_metric
from attribute_list import AttributeList
from class_list import ClassList
from bining import BinStructure


class TGBoost(object):
    """
    Tiny Gradient Boosting
    """
    def __init__(self):
        self.trees = []
        self.eta = None
        self.num_boost_round = None
        self.first_round_pred = None
        self.loss = None
        self.max_depth = None
        self.subsample = None
        self.colsample = None
        self.reg_lambda = None
        self.min_sample_split = None
        self.gamma = None
        self.num_thread = None
        self.min_child_weight = None
        self.scale_pos_weight = None

    def fit(self,
            features,
            label,
            validation_data=(None,None),
            early_stopping_rounds=np.inf,
            maximize=True,
            eval_metric=None,
            loss="logisticloss",
            eta=0.3,
            num_boost_round=1000,
            max_depth=6,
            scale_pos_weight=1,
            subsample=0.8,
            colsample=0.8,
            min_child_weight=1,
            min_sample_split=10,
            reg_lambda=1.0,
            gamma=0,
            num_thread=-1):

        """
        :param features: np.array
        :param label: np.array
        :param eta: learning rate
        :param num_boost_round: number of boosting round
        :param max_depth: max depth of each tree
        :param subsample: row sample rate when building a tree
        :param colsample: column sample rate when building a tree
        :param min_sample_split: min number of samples in a leaf node
        :param loss: loss object
                     logisticloss,squareloss, or customize loss
        :param reg_lambda: lambda
        :param gamma: gamma
        :param num_thread: number of threself.tree_predict_Xad to parallel
        :param eval_metric: evaluation metric, provided: "accuracy"
        """
        self.eta = eta
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample = colsample
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_sample_split = min_sample_split
        self.num_thread = num_thread
        self.eval_metric = eval_metric
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.first_round_pred = 0.0

        bin_structure = BinStructure(features)
        attribute_list = AttributeList(features, bin_structure)
        class_list = ClassList(label)

        # initial loss function
        if loss == "logisticloss":
            self.loss = LogisticLoss()
        elif loss == "squareloss":
            self.loss = SquareLoss()
            self.first_round_pred = label.mean()
        else:
            try:
                self.loss = CustomizeLoss(loss)
            except:
                raise NotImplementedError("loss should be 'logisticloss','squareloss', or customize loss function")

        # to evaluate on validation set and conduct early stopping
        # we should get (val_features,val_label)
        # and set some variable to check when to stop
        do_validation = True
        if not isinstance(validation_data, tuple):
            raise TypeError("validation_data should be (val_features, val_label)")

        val_features, val_label = validation_data
        val_pred = None
        if val_features is None or val_label is None:
            do_validation = False
        else:
            val_pred = np.ones(val_label.shape) * self.first_round_pred

        if maximize:
            best_val_metric = - np.inf
            best_round = 0
            become_worse_round = 0
        else:
            best_val_metric = np.inf
            best_round = 0
            become_worse_round = 0



        for i in range(self.num_boost_round):
            # weighted grad and hess
            Y.grad = Y.grad * Y.sample_weight
            Y.hess = Y.hess * Y.sample_weight
            # row and column sample before training the current tree
            data = X.sample(frac=self.colsample_bytree, axis=1)
            data = pd.concat([data, Y], axis=1)
            data = data.sample(frac=self.subsample, axis=0)
            Y_selected = data[['label', 'y_pred', 'grad', 'hess']]
            X_selected = data.drop(['label', 'y_pred', 'grad', 'hess', 'sample_weight'], axis=1)

            # train current tree
            tree = Tree()
            tree.fit(X_selected, Y_selected, max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                     colsample_bylevel=self.colsample_bylevel, min_sample_split=self.min_sample_split,
                     reg_lambda=self.reg_lambda, gamma=self.gamma, num_thread=self.num_thread)

            # predict the whole trainset and update y_pred,grad,hess
            preds = tree.predict(X)
            Y['y_pred'] += self.eta * preds
            Y['grad'] = self.loss.grad(Y.y_pred.values, Y.label.values)
            Y['hess'] = self.loss.hess(Y.y_pred.values, Y.label.values)

            # update feature importance
            for k in tree.feature_importance.iterkeys():
                self.feature_importance[k] += tree.feature_importance[k]

            self.trees.append(tree)

            # print training information
            if self.eval_metric is None:
                print "TGBoost round {iteration}".format(iteration=i)
            else:
                try:
                    mertric_func = get_metric(self.eval_metric)
                except:
                    raise NotImplementedError("The given eval_metric is not provided")

                train_metric = mertric_func(self.loss.transform(Y.y_pred.values), Y.label.values)

                if not do_validation:
                    print "TGBoost round {iteration}, train-{eval_metric} is {train_metric}".format(
                        iteration=i, eval_metric=self.eval_metric, train_metric=train_metric)
                else:
                    val_Y['y_pred'] += self.eta * tree.predict(val_X)
                    val_metric = mertric_func(self.loss.transform(val_Y.y_pred.values), val_Y.label.values)
                    print "TGBoost round {iteration}, train-{eval_metric} is {train_metric}, val-{eval_metric} is {val_metric}".format(
                        iteration=i, eval_metric=self.eval_metric, train_metric=train_metric, val_metric=val_metric
                    )

                    # check if to early stop
                    if maximize:
                        if val_metric > best_val_metric:
                            best_val_metric = val_metric
                            best_round = i
                            become_worse_round = 0
                        else:
                            become_worse_round += 1
                        if become_worse_round > early_stopping_rounds:
                            print "TGBoost training Stop, best round is {best_round}, best {eval_metric} is {best_val_metric}".format(
                                best_round=best_round, eval_metric=eval_metric, best_val_metric=best_val_metric
                            )
                            break
                    else:
                        if val_metric < best_val_metric:
                            best_val_metric = val_metric
                            best_round = i
                            become_worse_round = 0
                        else:
                            become_worse_round += 1
                        if become_worse_round > early_stopping_rounds:
                            print "TGBoost training Stop, best round is {best_round}, best val-{eval_metric} is {best_val_metric}".format(
                                best_round=best_round, eval_metric=eval_metric, best_val_metric=best_val_metric
                            )
                            break

    def predict(self, X):
        assert len(self.trees) > 0

        # TODO: actually the tree prediction can be parallel
        preds = np.zeros((X.shape[0],))
        preds += self.first_round_pred
        for tree in self.trees:
            preds += self.eta * tree.predict(X)

        return self.loss.transform(preds)