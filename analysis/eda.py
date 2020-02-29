"""
    这个文件的作用是做EDA的，它主要做以下几件事情：
    1. 查看数据集中，哪些是分类型特征，哪些是数值型特征
    2. 查看数据集中的缺失情况来决定用什么方法来补全数据集
"""

from utils import *
from tabulate import tabulate

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
    result.sort_values(by=["missing rate"], axis=0, ascending=False, inplace=True)
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
    save_path = os.path.join(dir_path, data_set.name + "_analysis.md")
    with open(save_path, "w") as f:  # Won't continue to append when rerunning
        for info_key, info_value in information_dict.items():
            f.write("# " + info_key + "\n")
            table = tabulate(info_value, headers="keys", tablefmt="pipe")
            f.write(table)
            f.write("\n\n")


if __name__ == "__main__":
    SetLogger("../log")

    # Get test set, train set and its target values
    logging.info("Load data set!")
    df_train_set, df_test_set = GetDataSet("../data")
    train_y = df_train_set["flag"].values
    df_train_set.drop("flag", axis=1, inplace=True)
    df_test_set.drop("flag", axis=1, inplace=True)
    logging.info("Load data set done!")

    # Check missing conditions
    Save2Markdown(df_train_set, ".")
    Save2Markdown(df_test_set, ".")

