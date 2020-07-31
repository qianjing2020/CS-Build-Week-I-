import numpy as np
import pandas as pd
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree import DecisionTree 
import time

# this file compares our decision tree with the DecisionTreeClassifier from sklearn library
class DataFetcher:
    # a simple class to fetch data
    def __init__(self):
        self.name = None
        self.len = None
        self.feature = None

    def iris_data(self):
        """get the famous iris data """
        file = 'Build-Week1/data/iris.data'
        df = pd.read_csv(file, header = None, skiprows=0)
        header = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
        data = df.to_numpy()
        self.name = 'Iris Dataset'
        self.len = len(data)
        self.feature = len(data[0])-1
        return data, header

    def banknote_data(self):
        """get the banknote_authentication data """
        file = 'Build-Week1/data/data_banknote_authentication.txt'
        df = pd.read_csv(file, header = None, skiprows=0)
        header = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        data = df.to_numpy()
        self.name = 'Banknotes Dataset'
        self.len = len(data)
        self.feature = len(data[0])-1
        return data, header

datafetcher = DataFetcher()
data, header = datafetcher.iris_data()
#data, header = datafetcher.banknote_data()

# split data into train, test
train, test = train_test_split(data, test_size = 0.2, random_state = 42)

# use libary decision tree to fit data and predict
start = time.time()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train[:,:-1], train[:,-1])
sktree_predictions = clf.predict(test[:,:-1])
end = time.time()
score = accuracy_score(test[:,-1], sktree_predictions)
print(f'Time for sklearn decision tree to finish {datafetcher.name} ({datafetcher.feature} features, {datafetcher.len} observations) fitting and prediction is {end-start:.4f} seconds. Accuracy score is {score}.')
# tree.plot_tree(clf)

# use our decision tree to fit data and predict
# prepare data for my tree

start = time.time()
my_tree = DecisionTree()
my_tree.fit(header[:-1], train)
my_predictions = []
i = 0
for observation in test:
    predicted = my_tree.predict(my_tree.tree, observation)
    my_predictions.append(predicted)

end = time.time()
my_score = accuracy_score(test[:, -1], my_predictions)
print(f'Time for my decisioin tree to finish {datafetcher.name} ({datafetcher.feature} features, {datafetcher.len} observations) fitting and prediction is {end-start:.4f} seconds. Accuracy score is {my_score}.')
# my_tree.print_tree()
