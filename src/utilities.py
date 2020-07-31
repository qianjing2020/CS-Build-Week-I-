"""functions and classes used in decision tree algorithm"""

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

def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right banch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    Defined as the uncertainty of the starting node, minus the weighted impurity of two child nodes.
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)

class Criterion:
    """A Criterion is used to split a dataset into two branches.
    """
    def __init__(self, header, col, value):
        """col are col numbers, int
        value are feature values, str or int or float
        """
        self.header = header
        self.col = col
        self.value = value

    def meet(self, observation):
        # Compare the feature value in an observation to the
        # feature value of the local criterion.
        observation_val = observation[self.col]
        if is_numeric(observation_val):
            # compare numerical feature
            return observation_val >= self.value
        else:
            # compare categorical feature
            return observation_val == self.value

    def __repr__(self):
        # Print the criterion.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (self.header[self.col], condition, str(self.value))

def split(data, criterion):
    """splits a dataset into two subset. 
    Loop through every observation, if it meets the criterion,
    add it to 'right_data', otherwise, add it to 'left_data'.
    """
    left_data, right_data = [], []
    for observation in data:
        if criterion.meet(observation):
            right_data.append(observation)
        else:
            left_data.append(observation)
    return left_data, right_data

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

class Leaf:
    """A Leaf node holds a dictionary of {class: number of its occurrence}
    from the training data that reach this leaf.
    """
    def __init__(self, data):
        self.predictions = list(class_counts(data).keys())

class DecisionNode:
    """A decision node stores the criterion for splitting
    and two child branches
    Input: criterion, left, and right data
    """
    def __init__(self, criterion, left, right):
        self.criterion = criterion
        self.left_branch = left
        self.right_branch = right


def class_counts(data):
    """Counts how many classes/labels in input data."""
    counts = {}  # a dictionary of label: count.
    for observation in data:
        # The data should have label as the last col
        label = observation[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    """Return true when a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

