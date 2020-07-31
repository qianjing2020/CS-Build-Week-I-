import numpy as np
import pandas as pd
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree import DecisionTree 
import time

# this file compares our decision tree with the DecisionTreeClassifier from sklearn library

def get_iris_data():
    """get the famous iris data """
    file = 'Build-Week1/data/iris.data'
    df = pd.read_csv(file, header = None, skiprows=0)
    header = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    data = df.to_numpy()
    return df, header
data, header = get_iris_data()
# split data into train, test
train, test = train_test_split(data, test_size = 0.2, random_state = 42)

# use libary decision tree to fit data and predict
start = time.time()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train.iloc[:,:-1], train.iloc[:,-1])
sktree_predictions = clf.predict(test.iloc[:,:-1])
end = time.time()
score = accuracy_score(test.iloc[:,-1], sktree_predictions)
print(f'Time for sklearn decision tree to finish Iris dataset fitting and prediction is {end-start:.4f} seconds. Accuracy score is {score}')
# tree.plot_tree(clf)

# use our decision tree to fit data and predict
# prepare data for my tree
train = train.to_numpy()
test = test.to_numpy()

start = time.time()
my_tree = DecisionTree()
my_tree.fit(train)
my_predictions = []
i = 0
for observation in test:
    predicted = my_tree.predict(my_tree.tree, observation)
    my_predictions.append(predicted)

end = time.time()
my_score = accuracy_score(test[:, -1], my_predictions)
print(f'Time for my decisioin tree to finish Iris dataset fitting and prediction is {end-start:.4f} seconds. Accuracy score is {my_score}')
# my_tree.print_tree()