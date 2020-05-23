import pandas as pd
import numpy as np


def DataProcess(base, knowledge, money, year):
    base = pd.read_csv(base, encoding="utf-8")
    konwledge = pd.read_csv(knowledge, encoding="utf-8")
    money_raw = pd.read_csv(money, encoding="utf-8")
    year_raw = pd.read_csv(year, encoding="utf-8")

    money_list = []
    money_index_list = []

    for i in range(3):
        money_list.append(money_raw[money_raw["year"] == i + 2015])
        money_index_list.append(list(money_list[i].columns))
        for n in range(2, len(money_index_list[i])):
            money_index_list[i][n] = str(i +
                                         2015) + "_" + money_index_list[i][n]
        money_list[i].columns = money_index_list[i]
        money_list[i].drop(labels="year", axis=1, inplace=True)

    money = pd.merge(money_list[0], money_list[1], on="ID")
    money = pd.merge(money, money_list[2], on="ID")

    year_list = []
    year_index_list = []

    for i in range(3):
        year_list.append(year_raw[year_raw["year"] == i + 2015])
        year_index_list.append(list(year_list[i].columns))
        for n in range(2, len(year_index_list[i])):
            year_index_list[i][n] = str(i + 2015) + "_" + year_index_list[i][n]
        year_list[i].columns = year_index_list[i]
        year_list[i].drop(labels="year", axis=1, inplace=True)

    year = pd.merge(year_list[0], year_list[1], on="ID")
    year = pd.merge(year, year_list[2], on="ID")

    data = pd.merge(base, konwledge, on="ID")
    data = pd.merge(data, money, on="ID")
    data = pd.merge(data, year, on="ID")
    return data

