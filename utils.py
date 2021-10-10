import numpy
import pandas
from typing import List
from sklearn.metrics import f1_score, accuracy_score


def check_label_dist(y: numpy.ndarray, desc: str):
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

