import numpy as np
import pandas as pd

def format_3_years(data_set: pd.DataFrame):
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

    return data_frame

if __name__ == '__main__':
    # Read all test data set
    prefix = "../../row_data/test1/"
    base = pd.read_csv(prefix + "base_test_sum.csv", encoding="gbk")
    base.drop("flag", axis=1, inplace=True)
    knowledge = pd.read_csv(prefix + "knowledge_test_sum.csv", encoding="gbk")
    money_raw = pd.read_csv(prefix + "money_report_test_sum.csv", encoding="gbk")
    year_raw = pd.read_csv(prefix + "year_report_test_sum.csv", encoding="gbk")

    # Concat dat set
    one_year_data = pd.merge(base, knowledge, on="ID")
    # print(one_year_data)
    year_raw.drop(["ID", "year"], axis=1, inplace=True)
    three_year_data = pd.concat([money_raw, year_raw], axis=1)
    three_year_data = format_3_years(three_year_data)
    # print(three_year_data)

    # Get final data set
    final_data_set = pd.merge(one_year_data, three_year_data, on="ID")
    final_data_set.to_csv("final_fuck.csv", encoding="utf-8", index=False)




