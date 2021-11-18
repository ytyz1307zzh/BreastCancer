import pandas as pd
import numpy as np
from utils import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 160)


def main(opt):
    dataframe = pd.read_csv(opt.input)
    dataframe.drop('Unnamed: 32', axis=1, inplace=True)
    dataframe.drop('id', axis=1, inplace=True)
    print(dataframe.head())
    print("\n" + "*" * 20 + "Read data success." + "*" * 20 + "\n")
    total_size = len(dataframe)

    # Features
    x = dataframe.drop(columns=LABEL)
    print("Data size: ", total_size, ", shape ", np.shape(x))

    # feature correlation
    col_corr = feature_correlation(x, opt.corr_thres)
    if opt.remove_corr:
        remove_correlation(x, col_corr)

    x = x.to_numpy()
    if opt.data_norm:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)  # numpy array
    print("Final data size: ", total_size, ", shape ", np.shape(x))
    # Predicting Value
    y = dataframe['diagnosis'].to_numpy()

    # train_test_splitting of the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
    assert len(x_train) == len(y_train), len(x_test) == len(y_test)
    print("Training size: ", len(x_train), ", testing size: ", len(x_test))

    check_label_dist(y, "Total")
    check_label_dist(y_train, "Train")
    check_label_dist(y_test, "Test")

    RandomGuess(y_test)
    LR(x_train, y_train, x_test, y_test, opt.fea_importance)
    KNN(x_train, y_train, x_test, y_test)
    DecisionTree(x_train, y_train, x_test, y_test, opt.fea_importance)
    NaiveBayes(x_train, y_train, x_test, y_test)
    SVM_nonlinear(x_train, y_train, x_test, y_test)
    SVM_linear(x_train, y_train, x_test, y_test)
    RandomForest(x_train, y_train, x_test, y_test, opt.fea_importance)
    Adaboost(x_train, y_train, x_test, y_test)
    NeuralNetwork(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default="data/data.csv", help="input data csv file")
    parser.add_argument('-data_norm', action="store_true", default=False,
                        help="specify to apply data normalization")
    parser.add_argument('-fea_importance', action='store_true', default=False,
                        help='specify to output feature importance')
    parser.add_argument('-corr_thres', type=float, default=0.8, help="threshold of feature correlation")
    parser.add_argument('-remove_corr', action='store_true', default=False,
                        help='specify to remove the correlated features')
    opt = parser.parse_args()
    main(opt)
