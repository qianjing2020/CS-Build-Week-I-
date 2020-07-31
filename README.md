---
layout: post
title: Build a decision tree using only Python
image: /img/tree.png
---

Tree tree tree! Don't we all love trees! Decision tree is a great scientific tool for classification and regression. This blog demonstrates how to make a decision tree classifier using basic Python code without extended libraries.  Our completed DecisionTree class implements **fit**, **predict** , and **print** methods. To skip the following blah blah part which took me hours to write up, you can directly go to [my github repo](https://github.com/qianjing2020/CS-Build-Week-I-/tree/master/src) for the code. 

![winter tree](https://www.texastreetrimmers.com/wp-content/uploads/2015/12/Frozen-Tree-in-Winter.jpg)     

Before even start building a decision tree, we need to know to what extent our data is still mixed after splitting at a node. To measure such "mixedness" or "impurity", we use metrics such as Gini impurity and Entropy. The two metrics have different formula of course but their performance are very similar. Since Gini impurity is more efficient to compute, we'll use it as our metrics to evaluate performance of splitting. Fun fact: Gini impurity is not an acronym, it is named after mathematician Corrado Gini.  

You can read more about [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) somewhere else. The Gini impurity is a function of probabilities of each class, and for a set of observations with J classes is calculated as follows: 
$$I_G(p) = 1 - \sum^J_{i=1} {p_i}^{2}$$

where $J$ = the number of all classes, and $p_i$ = the fraction of observations labeled with class $i$.

Let's start with simple example. Suppose we have a task to identify items from images, and the items have been boiled down to numbers and/or categories. We want to guess which class the item belongs with input of its features such as color, shape, and size. We'll call each row of the data table an observation, column 0 to column 2 are features, and the last column is label/class. We want to use a set of pre-classified items to build a decision tree so that we can predict the classes for new observations. This is obviously a supervised classification problem. 

| color    |  shape     |  size  |  class   |
| :-----:  |  :-----:   | :-----:|  :-----:  |    
|'Green'   |  'triangle'|  2   |  'Leaf'     |  
|  'Blue'  |  'rectangle' |  10  |  'Sky'      | 
|  'Green'  |  'oval'   |  1  |  'Leaf'   |  

By looking at the data, we can see that if we create the first split of the decision tree by comparing the color (feature in column 0), we can spit the items into two distinct classes: Leaf and Sky. The occurrence of Leaf as a class is $2/3$, and Sky $1/3$. The Gini impurity at this node is:
```I_G = 1 - (2/3)^2 - (1/3)^2 = 0.444```. As we recall, impurity measures how mixed our data is after the splitting.  If there is only one class exist in our data, then ```I_G = 1-1^2 = 0$```, and there is 0 mix/impurity in our data. Gini impurity lies between 0 and 1.

Below is a python function to calculate Gini impurity.
~~~python
def gini(data):
    """Calculate the Gini Impurity Score,
    meauring how mixed the data is.
    """
    counts = class_counts(data)
    impurity = 1
    for item in counts:
        # go over each label in counts
        prob = counts[item] / float(len(data))
        impurity -= prob**2
    return impurity
~~~
Based on Gini impurity, we can come up with another metrics to measure the "purity gain" for each splitting node. Information Gain of a node is defined as the impurities of node minus the weighted sum of the impurities of its two child nodes. 

~~~python
def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right banch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    Defined as the uncertainty of the starting node, minus the weighted impurity of two child nodes.
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)~~~

The next step to build a decision tree is to establish criterion to split the data into two branches. We create a **decision node** for such a split. A node will contain  and the associated criterion to split the data. From the very first node on the top, each node will have two child nodes down until no further split can happen. These nodes at the very bottom are called a **leaf nodes**, and they are simply the set of the classes appeared in training data. 

The tricky question to build such a decision tree is: what criterion do we apply and where are their respective locations in the tree? For the simple dataset in the above table, if the first criterion is ```color == "Blue"```, we can split the data into a **true branch** (the observations have color blue) and a **false branch** (the observations with color other than blue). Obviously the split and information gain will be quite different if we choose size for the first criterion. In addition, ```size >= 2``` and ```size >= 8``` will results in different tree structures. In fact, which feature we use and the feature value we choose to express the criterion will determine the split and the overall appereance of our final tree.  

Gladly we already have the tools (impurity metrics and Python) to find out what will be the right criterion at each node. Using training data, we can loop through all features and theirs observed values to find the best criterion, which in turn will results the greatest information gain. So our beautiful tree can have the right split at the right place and bud out leaves at the end. The algorithm for finding the best criterion is as follows.

~~~python
def find_best_split(header, data):
    """
    Input the data before before splitting.
    Find the best criterion for splitting based on greatest information gain. 
    All features and their unique values in data are considered when calculating the gain.
    Return the greatest gain and correspondent criterion 
    """
    best_gain = 0  # initialize the best information gain
    best_criterion = None  # initialize the criterion associated with the best gain
    current_impurity = gini(data) # the gini impurity before splitting
    n_features = len(data[0]) - 1  # number of features

    for col in range(n_features):  
        # col number for each feature
        feature_values = set([observation[col] for observation in data])  
        # unique values in the col
        
        for feature_val in feature_values:  # for each value
            criterion = Criterion(header, col, feature_val)
            # splitting the dataset
            left_data, right_data = split(data, criterion)
            # Skip this split if it doesn't divide the dataset.
            if len(left_data) == 0 or len(right_data) == 0:
                continue # skip the rest and goes to next feature_val
            
            # Calculate the information gain from this split
            gain = info_gain(left_data, right_data, current_impurity)

            # Update the best_gain if new gain value is greater
            if gain >= best_gain:
                best_gain, best_criterion = gain, criterion

    return best_gain, best_criterion
~~~

The next, we can define a split function that splits the dataset according to a criterion and hence create a node. Then we need to make two classes to implement the two basic node types. The decision nodes that contain criterion and child branches are instance of the **DecisionNode class**. The leaf nodes are instance of our **Leaf class**.  All these function and classes are in the [utilities](https://github.com/qianjing2020/CS-Build-Week-I-/blob/master/src/utilities.py) module. 

Now we are ready to code up our decision tree classifier. Our [decision tree code](https://github.com/qianjing2020/CS-Build-Week-I-/blob/master/src/decision_tree.py) imports the classes and functions from our utilities module, implements **fit** and **predict** methods. 

~~~python
class DecisionTree:
    """A binary decision tree gitto predict categorical classes"""
    def __init__(self):
        self.tree = None
        
    def fit(self, features, data):
        """ fit data to tree model using recursion"""
        # get best criterion lead to greatest infomation gain"""
        gain, criterion = find_best_split(features, data)
        # Base case: no further info gain, data reach leaf nodes, 
        if gain == 0:
            return Leaf(data)
        # split the data at the criterion
        left_data, right_data = split(data, criterion)
        # Recursively build the branches
        left_branch = self.fit(features, left_data)
        right_branch = self.fit(features, right_data)
        # Return a Decision Node, which contains the criterion and two child branches of the node
        fitted_node = DecisionNode(criterion, left_branch, right_branch)    
        self.tree = fitted_node    
        return fitted_node

    def predict(self, node, observation):
        """Classify the data to either branch until become a leaf."""
        # Base case: reached a leaf
        if isinstance(node, Leaf):
            return node.predictions
        # recursive case:
        if node.criterion.meet(observation):
            return self.predict(node.right_branch, observation)
        else:
            return self.predict(node.left_branch, observation)
            
~~~

Code testing: The python code includes the following make-up tiny dataset to play with. Running our [decision tree code](https://github.com/qianjing2020/CS-Build-Week-I-/blob/master/src/decision_tree.py), we can create an DecisionTree class instance, fit it with the make-up training data, and predict the outcome for new observations.

| color    |  shape     |  size  |  class   |
| :-----:  |  :-----:   | :-----:|  :-----:  |    
|'Green'   |  'triangle'|  2   |  'Leaf'     |  
|  'Blue'  |  'polygon' |  10  |  'Sky'      | 
|  'Red'   |  'round'   |  8   |  'Ballon'   |  
|  'Red'   |  'polygon' |  1   |  'Flower'   |  
|  'White' |  'round'   |  1   |  'Flower'   |  
|  'Green' |  'polygon' |  10  |  'Meadow'   | 


```python
dt = DecisionTree()
dt.fit(training_data)
dt.print_tree()
dt.predict(dt.tree, observation)
```
You will have the following output running [decision_tree.py](https://github.com/qianjing2020/CS-Build-Week-I-/blob/master/src/decision_tree.py).  

~~~python
Is size >= 2?
 -> False:
  Predict ['Flower']
 -> True:
  Is size >= 8?
  -> False:
   Predict ['Leaf']
  -> True:
   Is shape == round?
   -> False:
    Is color == Blue?
    -> False:
     Predict ['Meadow']
    -> True:
     Predict ['Sky']
   -> True:
    Predict ['Ballon']
Actual: Leaf, predicted: ['Leaf']
Actual: Flower, predicted: ['Flower']
Actual: Flower, predicted: ['Leaf']
Actual: Ballon, predicted: ['Ballon']
Actual: Meadow, predicted: ['Meadow']
~~~

In terms of performance, we can test our code against the [ DecisionTreeClassifier](https://scikit-learn.org/stable/modules/tree.html#tree-classification) from sklearn library. We used the famous [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). When the data is split into train, test data as 80%, 20%, respectively, the results of two decision trees are as follows:

~~~python
Time for sklearn decision tree to finish Iris Dataset (4 features, 150 observations) fitting and prediction is 0.0184 seconds. Accuracy score is 1.0.
Time for my decisioin tree to finish Iris Dataset (4 features, 150 observations) fitting and prediction is 0.0407 seconds. Accuracy score is 1.0.
~~~

When testing on larger dataset: [Banknote authentication dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication), both decision tree classifiers achieved same accuracy, but it took 100x longer time for our home-made code to finish running. 

~~~python
Time for sklearn decision tree to finish Banknotes Dataset (4 features, 1372 observations) fitting and prediction is 0.3358 seconds. Accuracy score is 0.9782.
Time for my decisioin tree to finish Banknotes Dataset (4 features, 1372 observations) fitting and prediction is 38.0418 seconds. Accuracy score is 0.9782.
~~~

To help get an idea of the time complexity of our decision tree, here is a rough estimation: 
Given $N$ observations and $K$ features, the depth of the tree would be $O(logN)$ to $O(N)$, the later being the worst scenario. Since finding the best criterion at a node requires testing all features and their values, this time complexity on each node can be $O(KN)$. Assume runtime for splitting the tree at each node is of $O(1)$, the time complexity for build the entire tree would be some where between $O(KNlogN)$ to $O(KN^2)$ . 

In conclusion, we built a basic decision tree classifier from scratch using Python Language. Its performance is very similar to the one in sklearn library. However, when data size increase to ~$10^3$, the runtime is a much longer than the sklearn decision tree classifier. I would like to dive deeper into the more efficient algorithms that sklearn provides (Thanks to open source!), but since this blog focuses on only understanding the mechanism behind the decision tree classifier, let's take a break! 

Related resources:

[Wikipedia: Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning)

[StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk&t=909s)

[Comparing these two impurity metrics](https://www.quora.com/What-is-difference-between-Gini-Impurity-and-Entropy-in-Decision-Tree#:~:text=in%20a%20node.-,Gini%3A%2D%20It%20shows%20that%20the%20data%20point%20that%20we%20have,small%20values%20show%20less%20impurity.)

[Write a Decision Tree Classifier from scratch](https://www.youtube.com/watch?v=LDRbO9a6XPU)

[Implement a Decision Tree Algorithm from scarth in Python](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/)

[Discussion on decision tree complexity](https://www.quora.com/How-do-I-calculate-the-time-complexity-of-a-decision-tree-machine-learning-algorithm)
