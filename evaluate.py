from utils import *
from main import model_path, models
from sklearn.metrics import auc, roc_curve, \
    accuracy_score, precision_score, recall_score

def GetAuc(model, X, Y):
    fpr, tpr, thresholds = roc_curve(Y, model.predict_proba(X)[:, 1])
    auc_score = auc(fpr, tpr)
    return auc_score

def GetConfusionMatrix(y_true, y_pred):
    confusion_matrix = {}
    # Get true count and false count
    total_count = len(y_pred)
    true_count = sum(y_pred)
    false_count = total_count - true_count
    # Get fpr, tpr, fnr, tnr
    fpc, tpc, tnc, fnc = 0, 0, 0, 0
    for y1, y2 in zip(y_true, y_pred):
        if y1 == 0 and y2 == 1:
            fpc += 1
        elif y1 == 1 and y2 == 1:
            tpc += 1
    tnc = false_count - fpc
    fnc = true_count - tpc
    # Assignment
    confusion_matrix["fpr"] = fpc / false_count
    confusion_matrix["tpr"] = tpc / true_count
    confusion_matrix["tnr"] = tnc / false_count
    confusion_matrix["fnr"] = fnc / true_count
    return confusion_matrix

def GetMetrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "F1": 2 * precision * recall / (precision + recall)
    }
    return metrics

if __name__ == "__main__":
    SetLogger("./log")

    # Get test set, train set and its target values
    logging.info("Loading data set..")
    df_train_set, df_test_set = GetDataSet("./data")
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

    model_instance = []
    for model in models:
        model_instance = LoadModel(model_path, model)
        model_instance.fit(df_train_set, train_y)
        y_pred = model_instance.predict(df_test_set)
        print("auc: " + str(GetAuc(model_instance, df_train_set, train_y)))
        print(GetMetrics(test_y, y_pred))
        print(GetConfusionMatrix(test_y, y_pred))
        print()





