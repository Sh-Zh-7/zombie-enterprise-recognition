from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

filterwarnings("ignore")

from plot import *
from utils import *

K = 5


def GetMatrices(prediction_list, ground_truth_list):
    tp, tn, fp, fn = 0, 0, 0, 0
    for prediction, ground_truth in zip(prediction_list, ground_truth_list):
        if prediction == 1:
            if ground_truth == 1:
                tp += 1
            else:
                fp += 1
        else:
            if ground_truth == 1:
                fn += 1
            else:
                tn += 1
    return tp, tn, fp, fn


def Train(model, X_train, y_train):
    # Initialize
    oob = 0  # Out-of-bag scores
    fprs, tprs, scores = [], [], []  # ROC curve
    feature_importance = pd.DataFrame(np.zeros((X_train.shape[1], K)),
                                      columns=["Fold_{}".format(i) for i in range(1, K + 1)])
    skf = StratifiedKFold(n_splits=K, random_state=K, shuffle=True)
    # Training
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print("Fold {}".format(fold))
        # Fitting model
        model.fit(X_train[trn_idx], y_train[trn_idx])
        # Computing train AUC score
        trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx],
                                                     model.predict_proba(X_train[trn_idx])[:, 1])
        trn_auc_score = auc(trn_fpr, trn_tpr)
        # Computing validation AUC score
        val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx],
                                                     model.predict_proba(X_train[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)
        # Append in list
        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)
        # Export Importance
        feature_importance.iloc[:, fold - 1] = model.feature_importances_
        # Out of bag score
        oob += model.oob_score_ / K
        print("Fold {} OOB Score: {}".format(fold, model.oob_score_))
        print("Average OOB Score: {}".format(oob))
    # Save model
    # joblib.dump(model, "./models/checkpoint.pkl")
    return fprs, tprs, scores, feature_importance


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

    # Using RandomForest
    params = Params("../models/RandomForest/best_params.json")
    rfc = RandomForestClassifier(**params.dict)
    # rfc.fit(df_train_set, train_y)
    fprs, tprs, scores, importance = Train(rfc, df_train_set.values, train_y)
    prediction = rfc.predict(df_test_set)

    PlotROCCurve(fprs, tprs)
    plt.show()

    tp, tn, fp, fn = GetMatrices(prediction, test_y)
    total = len(prediction)
    print("length: ", total)
    print("tp: ", tp)
    print("fp: ", fp)
    print("tn: ", tn)
    print("fn: ", fn)
    print("accuracy: ", (tp + tn) / total)

