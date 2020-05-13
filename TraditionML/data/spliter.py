"""
    这个:文件的作用就是把已经处理好的data_set.csv
    按照8:2的比例重新分为train_set.csv和test_set.csv
    理由如下：
        1. 如果直接在程序中使用train_test_split，那么会增加程序运行的时间，降低开发效率。
        2. 从物理上直接把文件分成两个文件，能有效避免data leakage，避免我们的估计过于乐观
"""

import random
import pandas as pd
import logging
import utils

SPILT_RATE = 0.2

def GetDataSet(filename: str):
    """ 读取整一个data_set文件，注意这个文件时utf-8编码格式的 """
    return pd.read_csv(filename, encoding="utf-8")

def GetRowCount(data_set: pd.DataFrame):
    """ 获取data_set的行数 """
    return len(data_set)

def GetRandomNumber(n: int):
    """ 从[0, n)范围中按照split_rate划分 """
    all_rows = range(n)
    test_rows = random.sample(range(n), int(n * SPILT_RATE))
    train_rows = [row for row in all_rows if row not in test_rows]
    return train_rows, test_rows

def ExportTrainAndTestSet(data_set: pd.DataFrame, train_rows, test_rows):
    """ 根据给定的训练集和测试集行数导出对应的train_set和test_set """
    data_set.iloc[train_rows].to_csv("train_set.csv", index=False)
    data_set.iloc[test_rows].to_csv("test_set.csv", index=False)


if __name__ == "__main__":
    utils.SetLogger("../log")
    data_set = GetDataSet("./data_set.csv")
    line_count = GetRowCount(data_set)
    train_lines, test_lines = GetRandomNumber(line_count)
    logging.info("Export train set and test set...")
    ExportTrainAndTestSet(data_set, train_lines, test_lines)
    logging.info("Done!")
