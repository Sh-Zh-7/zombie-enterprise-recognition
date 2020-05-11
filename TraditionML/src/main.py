from sklearn.svm import SVC

from src.utils import *

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.classifier import StackingClassifier

import warnings
warnings.filterwarnings("ignore")

SEED = 233

model_path = {
    AdaBoostClassifier: "../models/AdaBoost",
    KNeighborsClassifier: "../models/KNN",
    LogisticRegression: "../models/LogisticRegression",
    RandomForestClassifier: "../models/RandomForest",
    SVC: "../models/SVC"
}
models = [AdaBoostClassifier, KNeighborsClassifier, LogisticRegression, RandomForestClassifier]


def GetAccuracy(model, data_set, y_true):
    y_pred = model.predict(data_set)
    print(accuracy_score(y_true, y_pred))

if __name__ == "__main__":
    SetLogger("../log")

    # Get test set, train set and its target values
    logging.info("Loading data set..")
    df_train_set, df_test_set = GetDataSet("../data")
    # Get labels
    train_y = df_train_set["flag"].values
    df_train_set.drop("flag", axis=1, inplace=True)
    test_y = df_test_set["flag"].values
    df_test_set.drop("flag", axis=1, inplace=True)
    logging.info("Done!")

    # Dealing with missing values
    logging.info("Pre-processing..")
    DealWithMissingValues(df_train_set)
    DealWithMissingValues(df_test_set)
    df_train_set = FeatureEngineering(df_train_set)
    df_test_set = FeatureEngineering(df_test_set)
    logging.info("Done!")

    # Training
    # In mlens part, you don't need to train the base estimator one by one.
    model_instance = []
    for model in models:
        model_instance.append(LoadModel(model_path, model))
    # meta_model = SVC(kernel="rbf", degree=3, probability=True)
    meta_model = LogisticRegression()

    # Ensemble
    # ensemble = SuperLearner(scorer=accuracy_score, random_state=SEED)
    # ensemble.add(model_instance)
    # ensemble.add_meta(lightgbm.LGBMClassifier())
    # ensemble.fit(df_train_set, train_y)

    # ensemble = StackingAverageModel(model_instance, meta_model)
    # ensemble.fit(df_train_set.values, train_y)

    ensemble = StackingClassifier(classifiers=model_instance, meta_classifier=meta_model, use_probas=True)
    ensemble.fit(df_train_set.values, train_y)

    # Get Accuracy
    GetAccuracy(ensemble, df_test_set, test_y)
