def gini(data):
    """Calculate the Gini Impurity Score.
    """
    counts = class_counts(data)
    impurity = 1
    for lbl in counts:
        # go over each label in counts
        prob_of_lbl = counts[lbl] / float(len(data))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    """Information Gain is measured as the uncertainty of the starting 
    node, minus the weighted impurity of two child nodes.
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - w * gini(left) - (1 - w) * gini(right)


class Criterion:
    """A Criterion is used to split a dataset into two subsets.
        """
    def __init__(self, col, value):
        self.col = col
        self.value = value

    def meet(self, observation):
        # Compare the feature value in an observation to the
        # feature value in this criterion.
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
        return "Is col %s %s %s?" % (self.col, condition, str(self.value))

def split(data, criterion):
    """splits a dataset. Check each observation,  if it meetes the criterion,
        add it to 'true data', otherwise, add it to 'false data'.
    """
    true_data, false_data = [], []
    for observation in data:
        if criterion.meet(observation):
            true_data.append(observation)
        else:
            false_data.append(observation)
    return true_data, false_data

def find_best_split(data):
    """Find the best criterion at current node. All features and their unique values 
    are considered to calculate information gain, and the greatest gain and correspondent
    criterion will be returned"""
    best_gain = 0  # initialize the best information gain
    best_criterion = None  # initialize the criterion associated with the best gain
    current_uncertainty = gini(data) # the gini score at the tree stump
    n_features = len(data[0]) - 1  # number of features

    for col in range(n_features):  # for each feature
        feature_values = set([observation[col] for observation in data])  # unique values in the col
        
        for feature_val in feature_values:  # for each value
            criterion = Criterion(col, feature_val)
            # splitting the dataset
            true_data, false_data = split(data, criterion)
            # Skip this split if it doesn't divide the dataset.
            if len(true_data) == 0 or len(false_data) == 0:
                continue # skip the rest and goes to next feature_val
            # Calculate the information gain from this split
            gain = info_gain(true_data, false_data, current_uncertainty)

            # Update the best_gain if new gain value is greater
            if gain >= best_gain:
                best_gain, best_criterion = gain, criterion

    return best_gain, best_criterion

class Leaf:
    """A Leaf node holds a dictionary of {class: number of its occurrence}
    from the training data that reach this leaf.
    """
    def __init__(self, data):
        self.predictions = class_counts(data)

class DecisionNode:
    """At decision node stores the criterion for splitting
    and two child branches
    """
    def __init__(self, criterion, true_branch, false_branch):
        self.criterion = criterion
        self.true_branch = true_branch
        self.false_branch = false_branch


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

