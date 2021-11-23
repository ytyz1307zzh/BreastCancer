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
    # drop columns that are not features
    dataframe.drop('Unnamed: 32', axis=1, inplace=True)
    dataframe.drop('id', axis=1, inplace=True)
    print(dataframe.head())
    print("\n" + "*" * 20 + "Read data success." + "*" * 20 + "\n")
    total_size = len(dataframe)

    # Features
    x = dataframe.drop(columns=LABEL)
    print("Data size: ", total_size, ", shape ", np.shape(x))

    # feature correlation
    col_corr = feature_correlation(x, opt.corr_thres, opt.do_plot)
    if opt.remove_corr:
        remove_correlation(x, col_corr)

    # features and labels
    x = x.to_numpy()
    y = dataframe[LABEL].to_numpy()

    # univariate feature selection using chi2 requires non-negative feature values
    if opt.fea_select == 'univar':
        x = univariate_feature_select(x, y, opt.n_select)

    # data normalization
    if opt.data_norm:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)  # numpy array
    print("Final data size: ", total_size, ", shape ", np.shape(x))

    # feature selection
    if opt.fea_select == 'sfs':
        x = sequential_feature_select(x, y, opt.n_select)
    if opt.fea_select == 'rfecv':
        x = recursive_feature_elimination(x, y)

    # train_test_splitting of the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
    assert len(x_train) == len(y_train), len(x_test) == len(y_test)
    print("Training size: ", len(x_train), ", testing size: ", len(x_test))

    # output some stats
    check_label_dist(y, "Total")
    check_label_dist(y_train, "Train")
    check_label_dist(y_test, "Test")

    # run the models
    RandomGuess(y_test)
    LR(x_train, y_train, x_test, y_test, opt.fea_importance, opt.do_plot)
    KNN(x_train, y_train, x_test, y_test)
    DecisionTree(x_train, y_train, x_test, y_test, opt.fea_importance, opt.do_plot)
    NaiveBayes(x_train, y_train, x_test, y_test)
    SVM_nonlinear(x_train, y_train, x_test, y_test)
    SVM_linear(x_train, y_train, x_test, y_test)
    RandomForest(x_train, y_train, x_test, y_test, opt.fea_importance, opt.do_plot)
    Adaboost(x_train, y_train, x_test, y_test)
    NeuralNetwork(x_train, y_train, x_test, y_test)
    FM(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default="data/data.csv", help="input data csv file")
    parser.add_argument('-data_norm', action="store_true", default=False,
                        help="specify to apply data normalization")
    parser.add_argument('-fea_importance', action='store_true', default=False,
                        help='specify to output feature importance')
    parser.add_argument('-corr_thres', type=float, default=0.9, help="threshold of feature correlation")
    parser.add_argument('-remove_corr', action='store_true', default=False,
                        help='specify to remove the correlated features')
    parser.add_argument('-fea_select', default='none', choices=['none', 'sfs', 'univar', 'rfecv'],
                        help='whether to use feature selection. '
                             'none: do not use. '
                             'sfs: apply sequential feature selection (from mlxtend). '
                             'univar: apply univariate feature selection. '
                             'rfecv: recursive feature elimination with cross validation '
                             'and random forest classification')
    parser.add_argument('-n_select', default=10, type=int,
                        help='the number of features to select, if -fea_select is not \'none\' '
                             '(not applicable to rfecv)')
    parser.add_argument('-do_plot', default=False, action='store_true',
                        help='Specify to plot the figures in matplotlib and seaborn.'
                             'Otherwise, no figures will be drawn.')
    opt = parser.parse_args()
    main(opt)
