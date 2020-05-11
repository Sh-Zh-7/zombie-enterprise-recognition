import pandas as pd


def get_prediction(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    三行代码实现预测
    :param df1: 带有ID的data frame, 最后的prediction顺序和这个ID的顺序一致
    :param df2: 带有纳税总额的data frame, 只要根据三年的综合是否为0即可进行判断
    :return: 预测的结果
    """
    df2["纳税总额"].fillna(0, inplace=True)
    return [1 if sum(df2[row["ID"] == df2["ID"]]["纳税总额"]) == 0 else 0
            for index, row in df1.iterrows()]


def get_accuracy(ground_truth, prediction):
    """
    跳过ground_truth中的nan项，做一个准确率和false positive的计算
    :param ground_truth: 真实值，可能带有nan
    :param prediction: 预测值
    :return: false positive的数量和准确率
    """
    num_true_positive = 0
    num_false_positive = 0
    
    skip_nan_length = 0
    for index in range(len(prediction)):
        if not pd.isna(ground_truth[index]):
            skip_nan_length += 1
            if prediction[index] == ground_truth[index]:
                num_true_positive += 1
            else:
                num_false_positive += 1
    return num_true_positive / skip_nan_length, num_false_positive

def main(filename1, filename2):
    data_set1 = pd.read_csv(filename1, encoding="GBK")
    data_set2 = pd.read_csv(filename2, encoding="GBK")
    ground_truth = data_set1["flag"]

    prediction = get_prediction(data_set1, data_set2)
    accuracy, false_positive_num = get_accuracy(ground_truth, prediction)
    print("#False Positive: %d" % false_positive_num)
    print("Accuracy: %.2f%%" % (accuracy * 100))
            

if __name__ == "__main__":
    # 得到train set和test set的false positive和accuracy
    print("Train Set:")
    main("../row_data/base_train_sum.csv", "../row_data/year_report_train_sum.csv")
    print("--------------------------------------------")
    print("Validation Set:")
    main("../row_data/base_verify1.csv", "../row_data/year_report_verify1.csv")
