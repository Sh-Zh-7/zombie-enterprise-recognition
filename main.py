from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def FeatureEngineering(df):
    numerical_features, categorical_features = GetNumericalAndCategoricalFeatures(df)
    # Deal with numerical features
    df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])
    # Deal with categorical features
    df = GetDummies(df, categorical_features)
    return df

def LoadModel(path):
    params = Params(os.path.join(path, "params.json"))
    model = RandomForestClassifier(**params.dict)
    return model

def GetAccuracy(model, data_set, y_true):
    y_pred = model.predict(data_set)
    print(accuracy_score(y_true, y_pred))

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

    # Training
    logging.info("Start training..")
    clf = LoadModel("./models/random_forest")
    clf.fit(df_train_set, train_y)

    # Get Accuracy
    GetAccuracy(clf, df_test_set, test_y)








