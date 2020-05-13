from main import *

from sklearn.model_selection import GridSearchCV

def SearchParams(model, trn_x, trn_y):
    params_list_path = model_path[model]
    tuned_parameters = Params(os.path.join(params_list_path, "params_list.json")).dict
    clf = GridSearchCV(model(), tuned_parameters, cv=5)

    clf.fit(trn_x, trn_y)
    # Save best params
    with open(os.path.join(params_list_path, "best_params.json"), "w") as f:
        json.dump(clf.best_params_, f, indent=4)


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

    for model in models:
        SearchParams(model, df_train_set.values, train_y)
