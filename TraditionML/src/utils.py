import json
import logging
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
    data_set = pd.get_dummies(data_set, columns=categorical_features, drop_first=False)
    data_set.name = reserve_name
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
    df_train_set = pd.read_csv(os.path.join(path, "train_set.csv"))
    df_test_set = pd.read_csv(os.path.join(path, "test_set.csv"))
    # Filter
    df_train_set.drop("ID", axis=1, inplace=True)
    df_test_set.drop("ID", axis=1, inplace=True)
    # Assign name
    df_train_set.name = "train"
    df_test_set.name = "test"
    return df_train_set, df_test_set


def GetNumericalAndCategoricalFeatures(data_set: pd.DataFrame):
    """ As it's name suggests, do feature engineering. """
    # Get numerical and categorical condition
    numerical_condition = data_set.describe()  # Numerical data
    categorical_condition = data_set.describe(include=["O"])  # categorical data
    # Get what pandas think it is numerical or categorical features
    numerical_features = list(numerical_condition.columns)
    categorical_features = list(categorical_condition.columns)
    # Adjust the old features set
    categorical_features.append("注册时间")
    categorical_features.append("专利")
    categorical_features.append("著作权")
    categorical_features.append("商标")
    numerical_features = [numerical_feature for numerical_feature in numerical_features
                                            if numerical_feature not in ["注册时间", "专利", "著作权", "商标"]]
    return numerical_features, categorical_features

def FeatureEngineering(df):
    numerical_features, categorical_features = GetNumericalAndCategoricalFeatures(df)
    # Deal with numerical features
    df[numerical_features] = MinMaxScaler().fit_transform(df[numerical_features])
    # Deal with categorical features
    df = GetDummies(df, categorical_features)
    return df


def SetLogger(log_path):
    """ Decide which directory to log """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        # Add file handler
        file_handler = logging.FileHandler(os.path.join(log_path, "train.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)
        # Add stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def DealWithMissingValues(data_set: pd.DataFrame):
    """ Simply fill nan values """
    # 如果使用pad会有一个缺陷，那就是他只看前面元素的情况来填充这里的值
    # 然而如果第一个数据就有缺失值，他是无法填充的
    data_set.fillna(0, inplace=True)


def CheckAndMakeDir(path):
    if not os.path.exists(path):
        print("The {} doesn't exist! Make a new directory!".format(path))
        os.makedirs(path)

def LoadModel(model_path, model):
    path = model_path[model]
    params = Params(os.path.join(path, "best_params.json"))
    estimator = model(**params.dict)
    return estimator
