import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data_set_path = "./data/train_set/train_set_3_years.csv"
    data_set = pd.read_csv(data_set_path, encoding="gb18030", index_col=False)

    columns = ["ID"]
    for column in data_set.columns:
        if column != "ID" and column != "year":
            columns.append("2015_" + column)
            columns.append("2016_" + column)
            columns.append("2017_" + column)

    ID = list(set(data_set["ID"]))
    data = np.zeros((len(ID), len(columns)))
    data_frame = pd.DataFrame(data, columns=columns)
    data_frame["ID"] = ID

    for each_id in ID:
        for year in [2015, 2016, 2017]:
            in_data = data_set[(data_set["ID"] == each_id) & (data_set["year"] == year)]
            for column in in_data.columns:
                if column != "ID" and column != "year":
                    target_column = str(year) + "_" + column
                    try:
                        in_float = in_data.iloc[0][column]
                        data_frame.loc[data_frame["ID"] == each_id, target_column] = in_float
                    except IndexError:
                        break

    data_frame.to_csv("./train_set_3_years.csv", index=False)
