from main import *

from sklearn.model_selection import GridSearchCV

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
    
    # Search best params
    tuned_parameters = Params("./models/random_forest/params_list.json").dict
    rfc = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5)
    rfc.fit(df_train_set, train_y)

    # Save best params
    with open("./models/random_forest/best_params.json", "w") as f:
        json.dump(rfc.best_params_, f, indent=4)

