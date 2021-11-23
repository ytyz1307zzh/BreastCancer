import numpy as np
import pandas
from typing import List
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFECV, f_classif
from mlxtend.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
from matplotlib import cm
from Constant import *
import seaborn as sns
# sns.set(style="whitegrid")


def check_label_dist(y: np.ndarray, desc: str):
    total_size = len(y)
    ben_size = len([i for i in y if i == 'B'])
    mal_size = total_size - ben_size
    print(f"{desc} data: benign {ben_size} ({ben_size / total_size * 100:.2f}%), "
          f"malicious {mal_size} ({mal_size / total_size * 100:.2f}%)")


def cal_score(pred: List[str], gold: List[str]):
    assert len(pred) == len(gold)
    TP, FP, FN, TN = 0, 0, 0, 0
    precision, recall, ben_f1, mal_f1, acc = 0, 0, 0, 0, 0
    num_ben, num_mal = 0, 0

    for p, g in zip(pred, gold):
        if g == 'B' and p == 'B':
            TP += 1
            num_ben += 1
        elif g == 'B' and p == 'M':
            FN += 1
            num_ben += 1
        elif g == 'M' and p == 'B':
            FP += 1
            num_mal += 1
        elif g == 'M' and p == 'M':
            TN += 1
            num_mal += 1
        else:
            raise ValueError("Wrong gold or predicted label")

        if g == p:
            acc += 1

    if TP > 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        ben_f1 = precision * recall * 2 / (precision + recall)

    print(f"Benign - Precision: {round(precision*100, 2)}, Recall: {round(recall*100, 2)}, F1: {round(ben_f1*100, 2)}")

    precision, recall = 0, 0
    if TN > 0:
        precision = TN / (TN + FN)
        recall = TN / (TN + FP)
        mal_f1 = precision * recall * 2 / (precision + recall)

    print(f"Malicious - Precision: {round(precision*100, 2)}, Recall: {round(recall*100, 2)}, F1: {round(mal_f1*100, 2)}")

    acc = acc / len(pred)
    weighted_f1 = (num_ben * ben_f1 + num_mal * mal_f1) / len(pred)
    macro_f1 = (ben_f1 + mal_f1) / 2

    print(f"Accuracy: {round(acc*100, 2)}, Weighted F1: {round(weighted_f1*100, 2)}, Macro F1: {round(macro_f1*100, 2)}")


def sequential_feature_select(x: np.ndarray, y: np.ndarray, n_features):
    print("\n" + "*" * 20 + "Sequential Feature Selection" + "*" * 20)
    lr_model = LogisticRegression()
    sfs_model = SequentialFeatureSelector(lr_model,
                                          k_features=n_features,
                                          forward=False,
                                          floating=True,
                                          scoring='accuracy',
                                          cv=5)
    sfs_model = sfs_model.fit(x, y)
    selected_idx = sfs_model.k_feature_idx_
    selected_fea = [FEATURE_LIST[idx] for idx in selected_idx]
    print("Selected features: ", ', '.join(selected_fea))
    print()

    # use the selected features to form a new feature set
    x_sfs = sfs_model.transform(x)
    return x_sfs


def univariate_feature_select(x: np.ndarray, y: np.ndarray, n_features):
    print("\n" + "*" * 20 + "Univariate Feature Selection" + "*" * 20)
    ufs_model = SelectKBest(f_classif, k=n_features)
    ufs_model = ufs_model.fit(x, y)
    selected_idx = ufs_model.get_support(indices=True)
    selected_fea = [FEATURE_LIST[idx] for idx in selected_idx]
    print("Selected features: ", ', '.join(selected_fea))
    print()

    # use the selected features to form a new feature set
    x_ufs = ufs_model.transform(x)
    return x_ufs


def recursive_feature_elimination(x: np.ndarray, y: np.ndarray):
    print("\n" + "*" * 20 + "Recursive Feature Elimination" + "*" * 20)
    rf_model = RandomForestClassifier(random_state=1234)
    rfecv_model = RFECV(rf_model, cv=5, scoring='accuracy')
    rfecv_model = rfecv_model.fit(x, y)
    selected_idx = rfecv_model.get_support(indices=True)
    selected_fea = [FEATURE_LIST[idx] for idx in selected_idx]
    print("Optimal number of features: ", rfecv_model.n_features_)
    print("Selected features: ", ', '.join(selected_fea))
    print()

    # use the selected features to form a new feature set
    x_rfecv = rfecv_model.transform(x)
    return x_rfecv


def feature_correlation(df: pandas.DataFrame, threshold: float):
    # Feature correlation
    corr = df.corr().round(2)

    # Mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    _, _ = plt.subplots(figsize=(20, 20))  # Set figure size
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Define custom colormap

    # Draw the heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    plt.tight_layout()
    plt.show()

    col_corr = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                col_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    # sort according to absolute correlation score, in descending order
    col_corr = sorted(col_corr, key=lambda x: x[2], reverse=True)
    print("\n" + "*" * 20 + "Feature Correlation" + "*" * 20)
    print(f"{len(col_corr)} pairs of correlated features")
    print(col_corr)
    print()

    return col_corr


def remove_correlation(df: pandas.DataFrame, col_corr):
    drop_features = []
    for f1, f2, score in col_corr:
        drop_features.append(f2)
    df.drop(columns=drop_features, inplace=True)


def plot_feature_importance(importance, alg: str):
    assert len(FEATURE_LIST) == len(importance)

    bars = plt.bar(FEATURE_LIST, importance)
    viridis = cm.get_cmap('viridis', 12)
    for i in range(len(bars)):
        bars[i].set_facecolor(viridis(importance[i] / max(importance)))

    plt.title(f'Feature Importance ({alg})', fontsize=12)
    plt.xlabel('Features', fontsize=10)
    plt.ylabel('Importance', fontsize=10)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def debug_cal_score():
    pred = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    gold = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    pred = ['B' if i == 1 else 'M' for i in pred]
    gold = ['B' if i == 1 else 'M' for i in gold]
    cal_score(pred, gold)

    benign_f1 = f1_score(y_true=gold, y_pred=pred, pos_label='B')
    print("Benign F1: ", benign_f1)
    mal_f1 = f1_score(y_true=gold, y_pred=pred, pos_label='M')
    print("Malicious F1: ", mal_f1)
    weighted_f1 = f1_score(y_true=gold, y_pred=pred, average="weighted")
    print("Weighted F1: ", weighted_f1)
    macro_f1 = f1_score(y_true=gold, y_pred=pred, average="macro")
    print("Macro F1: ", macro_f1)
    acc = accuracy_score(y_true=gold, y_pred=pred)
    print("Accuracy: ", acc)


def debug():
    debug_cal_score()


if __name__ == "__main__":
    debug()
