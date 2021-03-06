# Breast Cancer Prediction

### Environment
All experiments are run on Linux with Python 3.7. Note that the `polylearn` package could
be problematic to install on Windows so Linux is the preferred OS.
```
pandas==0.24.2
numpy==1.16.4
scikit-learn==0.24.0
mlxtend==0.19.0
```
Besides, to make the factorization machine (FM) work, you also need to install `polylearn`. 
The following installation guide is adopted from [the polylearn github repo](https://github.com/scikit-learn-contrib/polylearn).
```
# install dependencies via pip
pip install numpy scipy scikit-learn nose
pip install sklearn-contrib-lightning


# or via conda
conda install numpy scipy scikit-learn nose
conda install -c conda-forge sklearn-contrib-lightning

# clone the git repo, build environment and install
git clone https://github.com/scikit-learn-contrib/polylearn.git
cd polylearn
python setup.py build
sudo python setup.py install
```
To making the figures work (using the `-do_plot` argument), you may also need:
```
matplotlib==3.3.0
seaborn==0.9.0
```

### Data
We leverage the Breast Cancer Wisconsin (Diagnostic) Dataset collected by the researchers and surgeons from 
the University of Wisconsin. One can also find the raw dataset from [UCI's machine learning repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

### Run
```bash
python main.py [-input file_name]
```
where `file_name` is set to `data/data.csv` by default. For other optional arguments, see the Features and Functions section below.

### Features and Functions
1. Data normalization. Use the argument `-data_norm` to enable.
2. Feature importance. Use the argument `-fea_importance` to enable. To plot additional
figures, specify the `-do_plot` argument.
3. Feature correlation. To plot an additional figure, specify the `-do_plot` argument.
Use the argument `-remove_corr` and `-corr_thres` to remove highly correlated features according to a specified threshold.
4. Feature selection. This means selecting a subset of "important" features for prediction.
The selection methods include univariate feature selection, sequential feature selection and recursive feature elimination. 
Use `-fea_select` argument to specify the type of feature selection (or disable it) and `-n_select` argument to specify 
the number of features to preserve.
5. Higher-order features. We use factorization machines (FM) to allow higher-order features
generated by interactions between the original first-order features.
6. A variety of classification models are used to predict the results from the dataset. These models include 
logistic regression, SVM (linear & non-linear kernels), k nearest neighbors, naive bayes, decision tree, random forest, 
Adaboost classifier (with decision tree as base) and neural network. We also perform a random guess baseline for comparison.
7. A variety of evaluation metrics are used to evaluate the models' performance. The metrics include PRF scores for
both classes, while total accuracy, weighted F1 and Macro F1 take both classes into consideration.

### Known Issues
1. There is a known bug that if one tends to first select a subset of features and then conduct the analysis of feature importance, 
that will trigger an error.
