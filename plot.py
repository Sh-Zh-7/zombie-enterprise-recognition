"""
My own plot code, only to plot more conveniently.
See more details on https://github.com/Sh-Zh-7/ML_plot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def PlotCount(data, x_label, y_label):
    """ Simply plot count plot """
    sns.countplot(data, linewidth=5, edgecolor=sns.color_palette("dark", 2), palette="bright")

    plt.xlabel(x_label, size=15, labelpad=15)
    plt.ylabel(y_label, size=15, labelpad=15)


def PlotDist(data_set, labels, feature):
    """
    Simply plot distribution map.
    Remember to Modify your axes's labels!!
    """
    categories = labels.unique()

    indexes1 = (labels == categories[0])
    indexes2 = (labels == categories[1])

    sns.distplot(data_set[feature][indexes1], label='XXX', hist=True, color='#e74c3c')
    sns.distplot(data_set[feature][indexes2], label='XXX', hist=True, color='#2ecc71')

    plt.legend(loc='upper right', prop={'size': 10})


def PlotHeatMapRectangle(data_set):
    """ Plot rectangle heat map by using seaborn module """
    sns.heatmap(data_set.corr(), annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})


def PlotHeatMapTriangle(data_set):
    """ Plot triangle heat map by using mlen module """
    from mlens.visualization import corrmat
    corrmat(data_set.corr(), inflate=False)


def PlotLearningCurve(estimator, X, y):
    """ Simply Plot learning curve """
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=None, train_sizes=np.linspace(0.05, 1, 20)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, "o-", color="b")
    plt.plot(train_sizes, test_scores_mean, "o-", color="r")

    plt.xlabel("train_times")
    plt.ylabel("scores")


def PlotImportance(features, importance):
    """ Simply plot importance map """
    data = pd.DataFrame({"Features": features, "Importance": importance})
    data.sort_values(by="Importance", kind="quicksort", ascending=False, inplace=True)
    sns.barplot(x="Importance", y="Features", data=data)

    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)


def PlotROCCurve(fprs, tprs):
    from sklearn.metrics import auc
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(15, 15))
    # Plotting ROC for each fold and computing AUC scores
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))
    # Plotting ROC for random guessing
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plotting the mean ROC
    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), lw=2,
            alpha=0.8)

    # Plotting the standard deviation around the mean ROC Curve
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')

    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title('ROC Curves of Folds', size=20, y=1.02)
    ax.legend(loc='lower right', prop={'size': 13})