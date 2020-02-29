import os
import json
import pandas as pd
from tabulate import tabulate

class Params:
    """ Load models's hyper parameters """
    def __init__(self, path):
        with open(path, "r") as f:
            self.__dict__.update(json.load(f))

    def load(self, path):
        with open(path, "r") as f:
            self.__dict__.update(json.load(f))

    def dump(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        return self.__dict__

# Press Ctrl+- to hide the detail of the functions
def GetDummies(data_set, categorical_features):
    """ Reserve the origin attribute while getting dummies """
    reserve_name = data_set.name
    reserve_trn_len = data_set.trn_len
    data_set = pd.get_dummies(data_set, columns=categorical_features, drop_first=True)
    data_set.name = reserve_name
    data_set.trn_len = reserve_trn_len
    return data_set

def ConcatDF(train_set, test_set):
    """
    Concatenate train set and test set,
    This may lead to data leakage, but we have to do that
    because some tricks suck as dummy code will be wrong
    """
    df_all = pd.concat([train_set, test_set], sort=True).reset_index(drop=True)
    df_all.trn_len = train_set.shape[0]
    return df_all

def DivideDF(df_all):
    """ Divide the data set that concatenated from train set and test set """
    return df_all.iloc[:df_all.trn_len], df_all.iloc[df_all.trn_len:]

def GetDataSet(path):
    """ Get train and test data set. """
    # Read csv files
    df_train_set = pd.read_csv(os.path.join(path, "train.csv"))
    df_test_set = pd.read_csv(os.path.join(path, "test.csv"))
    # Filter
    df_train_set.drop("Id", axis=1, inplace=True)
    df_test_set.drop("Id", axis=1, inplace=True)
    # Assign name
    df_train_set.name = "train"
    df_test_set.name = "test"
    # Filter还真的不能在这里做，不然返回的就是原生的DataFrame对象了
    return df_train_set, df_test_set

def DealWithMissingValues(data_set: pd.DataFrame):
    """ Simply fill nan values """
    data_set.fillna(method="pad", inplace=True)

# Actually, there is no missing value
# The NA data do have meanings
def GetMissingValues(data_set: pd.DataFrame):
    """ Show missing numbers if it exists """
    # Get missing features and missing line counts
    missing_features = []
    missing_line_counts = []
    for column in data_set.columns:
        missing_line_count = data_set[column].isnull().sum()
        if missing_line_count != 0:
            missing_features.append(column)
            missing_line_counts.append(missing_line_count)
    missing_rate = [item / data_set.shape[0] for item in missing_line_counts]
    # Create given data set
    result = pd.DataFrame({"features": missing_features,
                           "missing lines": missing_line_counts,
                           "missing rate": missing_rate},
                          dtype="int64")
    return result


def Save2Markdown(data_set, dir_path):
    # Get all information
    numerical_feature = data_set.describe()  # Numerical data
    categorical_feature = data_set.describe(include=["O"])  # categorical data
    missing_conditions = GetMissingValues(data_set)
    information_dict = {
        "Numerical feature": numerical_feature,
        "Categorical feature": categorical_feature,
        "Missing conditions": missing_conditions
    }
    # Save to relevant markdown files
    save_path = os.path.join(dir_path, data_set.name + "_set_analysis.md")
    with open(save_path, "w") as f:  # Won't continue to append when rerunning
        for info_key, info_value in information_dict.items():
            f.write("# " + info_key + "\n")
            table = tabulate(info_value, headers="keys", tablefmt="pipe")
            f.write(table)
            f.write("\n\n")

def CheckAndMakeDir(path):
    if not os.path.exists(path):
        print("The {} doesn't exist! Make a new directory!".format(path))
        os.makedirs(path)