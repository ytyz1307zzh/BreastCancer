from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from utils import cal_score
import random


def RandomGuess(y_test):
    print("\n" + "*" * 20 + "Using Random Guess." + "*" * 20 + "\n")
    pred = [random.choice(['B', 'M']) for _ in range(len(y_test))]
    cal_score(pred=pred, gold=y_test.tolist())


def LR(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using logistic regression." + "*" * 20 + "\n")
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


def KNN(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using K nearest neighbors." + "*" * 20 + "\n")
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


def DecisionTree(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using Decision Tree." + "*" * 20 + "\n")
    model = DecisionTreeClassifier(random_state=1234)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


def NaiveBayes(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using Naive Bayes." + "*" * 20 + "\n")
    model = GaussianNB()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


def SVM_nonlinear(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using SVM with non-linear kernel." + "*" * 20 + "\n")
    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


def SVM_linear(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using SVM with linear kernel." + "*" * 20 + "\n")
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


def RandomForest(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using Random Forest." + "*" * 20 + "\n")
    model = RandomForestClassifier(oob_score=True, random_state=1234)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


def Adaboost(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using Adaboost on decision tree." + "*" * 20 + "\n")
    base = DecisionTreeClassifier(random_state=1234)
    model = AdaBoostClassifier(base_estimator=base, n_estimators=100, random_state=1234)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


def NeuralNetwork(x_train, y_train, x_test, y_test):
    print("\n" + "*" * 20 + "Using Neural Network." + "*" * 20 + "\n")
    model = MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=32, max_iter=100,
                          random_state=1234, early_stopping=True, n_iter_no_change=10)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cal_score(pred=pred, gold=y_test.tolist())


