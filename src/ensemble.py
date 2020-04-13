import numpy as np
import xgboost
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from src.utils import *

SEED = 233

class StackingAverageModel(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self, base_models, meta_model, n_fold=5):
        self.base_models = base_models
        # The clone here is only for pretending data leakage
        self.base_models_clone = [[clone(base_model)] * n_fold for base_model in base_models]

        self.meta_model = meta_model
        self.n_fold = n_fold

    def fit(self, X, y):
        """ Fit the X and y by using out-of-fold prediction """
        # We always use stratified-k-fold in classification tasks
        # In regression tasks, we just simply use k-fold
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=SEED)
        out_of_fold_prediction = np.zeros((X.shape[0], len(self.base_models)))
        for i, base_model in enumerate(self.base_models):
            for j, (train_index, test_index) in enumerate(kf.split(X, y)):
                model = self.base_models_clone[i][j]
                model.fit(X[train_index], y[train_index])
                out_of_fold_prediction[test_index, i] = model.predict(X[test_index])
        self.meta_model.fit(out_of_fold_prediction, y)

    def predict(self, X):
        """ Make predictions on the whole base models """
        meta_features = np.column_stack([
            np.column_stack([base_model.predict_proba(X) for base_model in base_models]).mean(axis=1)
            for base_models in self.base_models_clone]
        )
        return self.meta_model.predict(meta_features)

    def save_params(self, path):
        """ Save hyper parameters, both base models and meta models """
        path = os.path.join(path, "check_point")
        CheckAndMakeDir(path)
        # Save base models's hyper parameters
        base_models_dir = os.path.join(path, "base_models_params")
        CheckAndMakeDir(base_models_dir)
        for i, base_model_clone in enumerate(self.base_models_clone):
            each_base_model_dir = os.path.join(base_models_dir, self.base_models_name[i])
            CheckAndMakeDir(each_base_model_dir)
            for j, model in enumerate(base_model_clone):
                final_save_path = os.path.join(each_base_model_dir, self.base_models_name[i] + "_" + str(j + 1))
                joblib.dump(model, final_save_path + ".pkl")
        # Save meta model's hyper parameters
        meta_model_dir = os.path.join(path, "meta_model_param")
        CheckAndMakeDir(meta_model_dir)
        # final_save_path = os.path.join(meta_model_dir, self.meta_model_name)
        # joblib.dump(self.meta_model, final_save_path + ".pkl")


def LoadModels(path):
    """ Load base models and meta model for model ensemble """
    # Get base models
    base_models = []
    # base_model_dir = os.path.join(path, "base_models")
    # Secondly deal with normal models
    base_models_name = ["AdaBoost", "RandomForest", "SVC"]
    base_models_cntr = [AdaBoostClassifier, RandomForestClassifier, SVC]
    for base_model_dir, cntr in zip(base_models_name, base_models_cntr):
        model_path = os.path.join(path, base_model_dir)
        base_model = cntr(**Params(os.path.join(model_path, "best_params.json")).dict)
        base_models.append(base_model)
    # Get the meta model
    # meta_model_path = os.path.join(path, "meta_model/xgboost.json")
    # meta_model = xgboost.XGBRegressor(**Params(meta_model_path).dict)
    meta_model = xgboost.XGBClassifier()

    return base_models, meta_model
