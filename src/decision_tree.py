from utilities import Leaf, DecisionNode, split, find_best_split, class_counts

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

    def print_node(self, node, indent= " "):
        """print the tree in a readable format"""
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print(indent + "Predict", node.predictions)
            return
        # Recursively print the rest of the tree
        # Print the criterion at node
        print(indent + str(node.criterion))
        # Print the left branch
        print(indent + '-> False:')
        self.print_node(node.left_branch,  indent+ " ")
        # Print the right branch
        print(indent + '-> True:')
        self.print_node(node.right_branch, indent+ " ")
        
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
    header = ["color", "shape", "size", "label"]
    features = header[:-1]
    dt = DecisionTree()
    dt.fit(features, training_data)

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
        print(f"Actual: {observation[-1]}, predicted: {predicted}")
