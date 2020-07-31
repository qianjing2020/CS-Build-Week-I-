from utilities import Leaf, DecisionNode, split, find_best_split, class_counts

class DecisionTree:
    """A binary decision tree to predict categorical classes"""
    def __init__(self):
        self.tree = None
        self.header = None

    def fit(self, data):
        """ fit data to model"""
        # get best criterion lead to greatest infomation gain"""
        gain, criterion = find_best_split(data)
        # Base case: reach leaf, no further info gain
        if gain == 0:
            return Leaf(data)
        # split the data at the criterion
        true_data, false_data = split(data, criterion)
        # Recursively build the branch.
        true_branch = self.fit(true_data)
        false_branch = self.fit(false_data)
        # Return a criterion node, which contains the best split strategy and rest of the tree.
        tree_node = DecisionNode(criterion, true_branch, false_branch)
        self.tree = tree_node
        return self.tree

    def predict(self, node, observation):
        """Classify the data to either branch."""
        # Base case: reached a leaf
        if isinstance(node, Leaf):
            return node.predictions
        # recursive case:
        if node.criterion.meet(observation):
            return self.predict(node.true_branch, observation)
        else:
            return self.predict(node.false_branch, observation)

    def print_node(self, node, indent= " "):
        """print the tree in a readable format"""
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print(indent + "Predict", node.predictions)
            return
        # Recursively print the rest of the tree
        # Print the criterion at node
        print(indent + str(node.criterion))
        # Print the true branch
        print(indent + '-> True:')
        self.print_node(node.true_branch, indent+ " ")
        # Print the false branch
        print(indent + '-> False:')
        self.print_node(node.false_branch,  indent+ " ")

    def print_leaf(self):
        """A nicer way to print the predictions at a leaf."""
        counts = class_counts()
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs     
    
    def print_tree(self):
        self.print_node(self.tree)
 
if __name__ == '__main__':
    training_data = [
        ['Green', 'triagle', 2, 'Leaf'],
        ['Blue', 'polygon', 10, 'Sky'],
        ['Red', 'round', 8, 'Ballon'],
        ['Red', 'polygon', 1, 'Flower'],
        ['White', 'round', 1, 'Flower'],
        ['Green', 'polygon', 10, 'Meadow']
    ]
    # Column labels.
    header = ["color", "size", "label"]
    
    dt = DecisionTree()
    dt.fit(training_data)
    dt.print_tree()
    # Evaluate
    testing_data = [
        ['Green', 'triangle', 3, 'Leaf'],
        ['Yellow', 'round', 1, 'Flower'],
        ['Red', 'round', 2, 'Flower'],
        ['Red', 'round', 9, 'Ballon'],
        ['Green', 'polygon', 12, 'Meadow'],
    ]
    for observation in testing_data:
        predicted = dt.predict(dt.tree, observation)
        print("Actual: %s, predicted: %s" % (observation[-1], predicted))
