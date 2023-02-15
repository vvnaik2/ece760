import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# CONSTANTS
FEATURE_1 = 'X1'
FEATURE_2 = 'X2'
CLASS_LABEL_COLUMN_NAME = 'Y'
COLUMN_NAMES = [FEATURE_1, FEATURE_2, CLASS_LABEL_COLUMN_NAME] # Assumption is all text files have only 2 features 

_id = 0
# Decision Tree Node where 'left' is the 'then' branch & 'right' is the 'else' branch of the parent 'predicate'
# If node is a 'leaf', then 'label' is set to the class label, else 'predicate' contains the predicate in string format
# '_id' represents the node id which is unique for each node. 'parent_link' comprises of the parent _id followed by 
# _left or _right indicating if the node is a left child or right child

class d3Node:
    def __init__(self, left, right, leaf, label, predicate, parent_link, feature, split_val):
        global _id
        self.left = left
        self.right = right
        self.leaf = leaf
        self.label = label
        self.predicate = predicate
        self._id = _id + 1
        self.parent_link = parent_link
        self.feature = feature
        self.split_val = split_val
        _id = _id + 1

def read_csv(file_name, column_names):
    return pd.read_csv(file_name, sep=' ', header=None, names=column_names)

# Tries all possible numbers in feature as potential split values
def determine_best_split_entropy_and_gain_ratio(dataset, feature_column, debug = False):
    print("Searching best split candidate for: " + str(feature_column))
    best_split = None
    max_entropy = None
    max_gain_ratio = None
    class_entropy = calculate_class_entropy(dataset)
    for split in dataset[feature_column]:
        conditional_entropy, feature_entropy = calculate_conditional_and_feature_entropy(dataset, feature_column, split)
        gain = class_entropy - conditional_entropy
        if feature_entropy == 0:
            gain_ratio = 0
        else:
            gain_ratio = gain/feature_entropy
        if best_split is None or gain_ratio > max_gain_ratio:
            best_split = split
            max_gain_ratio = gain_ratio
            max_entropy = conditional_entropy
        if debug == True:
            print("Split: " + str(split) + ", Gain Ratio: " + str(gain_ratio) + ", Information Gain: " + str(conditional_entropy))
    return best_split, max_entropy, max_gain_ratio

def calculate_class_entropy(dataset):
    positive_count = len(dataset[dataset[CLASS_LABEL_COLUMN_NAME] == 1])
    negative_count = len(dataset[dataset[CLASS_LABEL_COLUMN_NAME] == 0])
    return calculate_entropy(positive_count, negative_count, len(dataset))

def calculate_conditional_and_feature_entropy(dataset, feature_column, split_value):
    # predicate: dataset[featureColumn] >= splitValue
    else_dataset = dataset[dataset[feature_column] < split_value]
    else_positive_count = len(else_dataset[dataset[CLASS_LABEL_COLUMN_NAME] == 1])
    then_dataset = dataset[dataset[feature_column] >= split_value]
    then_positive_count = len(then_dataset[dataset[CLASS_LABEL_COLUMN_NAME] == 1])
    else_entropy = calculate_entropy(else_positive_count, len(else_dataset)-else_positive_count, len(else_dataset))
    then_entropy = calculate_entropy(then_positive_count, len(then_dataset)-then_positive_count, len(then_dataset))
    conditional_entropy = (len(else_dataset)/len(dataset)) * else_entropy + (len(then_dataset)/len(dataset)) * then_entropy
    feature_entropy = calculate_entropy(len(then_dataset), len(else_dataset), len(dataset))
    return conditional_entropy, feature_entropy

def calculate_entropy(pos_count, neg_count, total_count):
    if pos_count == 0 or neg_count == 0:
        return 0
    return -1 * ((pos_count/total_count) * math.log(pos_count/total_count, 2) + (neg_count/total_count) * math.log(neg_count/total_count, 2))

def predict_majority_class(dataset):
    positive_count = len(dataset[dataset[CLASS_LABEL_COLUMN_NAME] == 1])
    negative_count = len(dataset[dataset[CLASS_LABEL_COLUMN_NAME] == 0])
    return 1 if positive_count >= negative_count else 0

def split_node_on(feature_name, dataset, split):
    leftDataset = dataset[dataset[feature_name] >= split]
    rightDataset = dataset[dataset[feature_name] < split]
    predicate = str(feature_name) + " >= " + str(split)
    return leftDataset, rightDataset, predicate

def generate_d3(dataset, parent_link):
    X1_split, X1_entropy, X1_gain_ratio = determine_best_split_entropy_and_gain_ratio(dataset, FEATURE_1)
    print("Best X1_split: " + str(X1_split) + ", X1_entropy: " + str(X1_entropy) + ", X1_gain_ratio: " + str(X1_gain_ratio))
    X2_split, X2_entropy, X2_gain_ratio = determine_best_split_entropy_and_gain_ratio(dataset, FEATURE_2)
    print("Best X2_split: " + str(X2_split) + ", X2_entropy: " + str(X2_entropy) + ", X2_gain_ratio: " + str(X2_gain_ratio))

    # stopping criteria
    if len(dataset) == 0:
        print("Dataset size is 0... Stopping")
        return d3Node(None, None, True, 1, "", parent_link, None, None)
    elif X1_gain_ratio == 0 and X2_gain_ratio == 0:
        print("X1 and X2 gain ratio is 0. Finding majority class and stopping")
        return d3Node(None, None, True, predict_majority_class(dataset), "", parent_link, None, None)
    else:
        if X1_gain_ratio > X2_gain_ratio:
            print("X1_gain_ratio > X2_gain_ratio. Choosing X1 as split feature...")
            leftDataset, rightDataset, predicate = split_node_on(FEATURE_1, dataset, X1_split)
            node = d3Node(None, None, False, None, predicate, parent_link, FEATURE_1, X1_split)
            leftNode = generate_d3(leftDataset, str(node._id) + "_left")
            rightNode = generate_d3(rightDataset, str(node._id) + "_right")
            node.left = leftNode
            node.right = rightNode
            return node
        else:
            print("X2_gain_ratio > X1_gain_ratio. Choosing X2 as split feature...")
            leftDataset, rightDataset, predicate = split_node_on(FEATURE_2, dataset, X2_split)
            node = d3Node(None, None, False, None, predicate, parent_link, FEATURE_2, X2_split)
            leftNode = generate_d3(leftDataset, str(node._id) + "_left")
            rightNode = generate_d3(rightDataset, str(node._id) + "_right")
            node.left = leftNode
            node.right = rightNode
            return node

def print_node(node):
    if node.leaf is True:
        print("[Id: " + str(node._id) + ", label: " + str(node.label) + ", parent: " + str(node.parent_link) + "]", end = " ")
    else:
        print("[Id: " + str(node._id) + ", predicate: " + str(node.predicate) + ", parent: " + str(node.parent_link) + "]", end = " ")
# Tree is printed one level at a time. 

def printD3(d3_node):
    level = []
    level_no = 0
    level.append(d3_node)
    print("Level: " + str(level_no))
    while(len(level) > 0):
        nextLevel = []
        while(len(level) > 0):
            node = level.pop(0)
            print_node(node)
            if node.left is not None:
                nextLevel.append(node.left)
            if node.right is not None:
                nextLevel.append(node.right)
        level = nextLevel
        level_no = level_no + 1
        print("")
        if len(level) > 0:
            print("Level: " + str(level_no))

def print_boolean_expression(label, predicates):
    result = ""
    for pred in predicates:
        if len(result) == 0:
            result = result + pred
        else:
            result = result + " and " + pred
    result = result + " == " + str(label)
    print(result)

def fetch_boolean_expressions(node, boolean_stack):
    left_stack = boolean_stack.copy()
    right_stack = boolean_stack.copy()
    if node.leaf is False:
        if node.left is not None:
            left_stack.append(node.predicate)
            fetch_boolean_expressions(node.left, left_stack)
        if node.right is not None:
            right_stack.append("!(" + node.predicate + ")")
            fetch_boolean_expressions(node.right, right_stack)
    else:
        print_boolean_expression(node.label, boolean_stack)

def computeTestError(dataset, d3_node):
    result = dataset.copy()
    result['P'] = result.apply(lambda row: predict(d3_node, row[FEATURE_1], row[FEATURE_2]), axis=1)
    result['error'] = pd.Series.abs(result['Y'] - result['P'])
    error = result['error'].sum() / len(result)
    return error

def scatter_plot(dataset):
    dataset.plot(kind = 'scatter', x = 'X1', y = 'X2')

def predict(d3_node, x1, x2):
    node = d3_node
    while(node.leaf == False):
        if node.feature == FEATURE_1:
            if x1 >= node.split_val:
                node = node.left
            else:
                node = node.right
        else:
            if x2 >= node.split_val:
                node = node.left
            else:
                node = node.right
    return node.label

def decision_boundary(dataset, d3_node):
    result = dataset.copy()
    result['P'] = result.apply(lambda row: predict(d3_node, row[FEATURE_1], row[FEATURE_2]), axis=1)
    result.plot(kind = 'scatter', x = 'X1', y = 'X2', c = result['P'].apply(lambda p: 'red' if p == 1 else 'blue'))

def plot_decision_boundary(dataset, d3_node, X1_lines, X2_lines):
    result = dataset.copy()
    result['P'] = result.apply(lambda row: predict(d3_node, row[FEATURE_1], row[FEATURE_2]), axis=1)
    fig, ax = plt.subplots()
    [ax.axhline(y=i, linestyle='-') for i in X2_lines]
    [ax.axvline(x=i, linestyle='-') for i in X1_lines]
    [ax.scatter(x = result[FEATURE_1], y = result[FEATURE_2], c = result['P'].apply(lambda p: 'red' if p == 1 else 'blue') )]

def fetch_decision_lines(d3_node):
    X1=[]
    X2=[]
    level = []
    level.append(d3_node)
    while(len(level) > 0):
        nextLevel = []
        while(len(level) > 0):
            node = level.pop(0)
            if node.leaf is False:
                if node.feature == FEATURE_1:
                    X1.append(node.split_val)
                else:
                    X2.append(node.split_val)
            if node.left is not None:
                nextLevel.append(node.left)
            if node.right is not None:
                nextLevel.append(node.right)
        level = nextLevel
    return X1, X2

def get_number_of_nodes(root):
    count = 0
    level = []
    level.append(root)
    count = count + 1
    while(len(level) > 0):
        nextLevel = []
        while(len(level) > 0):
            node = level.pop(0)
            if node.left is not None:
                nextLevel.append(node.left)
                count = count + 1
            if node.right is not None:
                nextLevel.append(node.right)
                count = count + 1
        level = nextLevel
    return count
 




#data = [[1,0,1],[0,1,1],[1,1,0],[0,0,0]]
#d2_dataset = pd.DataFrame(data, columns=['X1','X2','Y'])
#q2_d3 = generate_d3(d2_dataset,'root')
#printD3(q2_d3)
#decision_boundary(d2_dataset,q2_d3)

## D1.txt
#D1_dataset = read_csv(file_name = 'D2.txt', column_names = COLUMN_NAMES)
#global _id
#_id = 0
#d3_root_D1 = generate_d3(D1_dataset, "root")
#printD3(d3_root_D1)

#scatter_plot(D1_dataset)
#decision_boundary(D1_dataset, d3_root_D1)
#X1_lines, X2_lines = fetch_decision_lines(d3_root_D1)
#plot_decision_boundary(D1_dataset, d3_root_D1, X1_lines, X2_lines)



#n = []
#err = []
#Dbig_dataset = read_csv(file_name = 'Dbig.txt', column_names = COLUMN_NAMES)
#Dbig_permutation = Dbig_dataset.iloc[np.random.RandomState(seed = 5).permutation(len(Dbig_dataset))].reset_index(drop=True)
#Dbig_train = Dbig_permutation[0:8192]
#Dbig_test = Dbig_permutation[8192:]
#D32 = Dbig_train[0:32]
#D128 = Dbig_train[:128]
#D512 = Dbig_train[:512]
#D2048 = Dbig_train[:2048]
#D8192 = Dbig_train[:]
#global _id
#_id = 0
#d32_d3 = generate_d3(D32, 'root_d32')
#err.append(computeTestError(Dbig_test, d32_d3))
#computeTestError(Dbig_test, d32_d3)
#decision_boundary(Dbig_permutation, d32_d3)
#X1_lines, X2_lines = fetch_decision_lines(d32_d3)
#plot_decision_boundary(Dbig_permutation, d32_d3, X1_lines, X2_lines)
#n.append(get_number_of_nodes(d32_d3))
#get_number_of_nodes(d32_d3)
#
#print("Printing n and errn lists respectively:")
#print(n)
#print(err)
#plt.plot(n,err)
#
#d32_d3 = generate_d3(D128, 'root_d32')
#err.append(computeTestError(Dbig_test, d32_d3))
##computeTestError(Dbig_test, d32_d3)
##decision_boundary(Dbig_permutation, d32_d3)
##X1_lines, X2_lines = fetch_decision_lines(d32_d3)
##plot_decision_boundary(Dbig_permutation, d32_d3, X1_lines, X2_lines)
#n.append(get_number_of_nodes(d32_d3))
#get_number_of_nodes(d32_d3)
#print("Printing n and errn lists respectively:")
#print(n)
#print(err)
#plt.plot(n,err)
#
#d32_d3 = generate_d3(D512, 'root_d32')
#err.append(computeTestError(Dbig_test, d32_d3))
##computeTestError(Dbig_test, d32_d3)
##decision_boundary(Dbig_permutation, d32_d3)
##X1_lines, X2_lines = fetch_decision_lines(d32_d3)
##plot_decision_boundary(Dbig_permutation, d32_d3, X1_lines, X2_lines)
#n.append(get_number_of_nodes(d32_d3))
#get_number_of_nodes(d32_d3)
#print("Printing n and errn lists respectively:")
#print(n)
#print(err)
#plt.plot(n,err)
#
#d32_d3 = generate_d3(D2048, 'root_d32')
#err.append(computeTestError(Dbig_test, d32_d3))
##computeTestError(Dbig_test, d32_d3)
##decision_boundary(Dbig_permutation, d32_d3)
##X1_lines, X2_lines = fetch_decision_lines(d32_d3)
##plot_decision_boundary(Dbig_permutation, d32_d3, X1_lines, X2_lines)
#n.append(get_number_of_nodes(d32_d3))
#get_number_of_nodes(d32_d3)
#print("Printing n and errn lists respectively:")
#print(n)
#print(err)
#plt.plot(n,err)
#
#d32_d3 = generate_d3(D8192, 'root_d32')
#err.append(computeTestError(Dbig_test, d32_d3))
#computeTestError(Dbig_test, d32_d3)
#decision_boundary(Dbig_permutation, d32_d3)
#X1_lines, X2_lines = fetch_decision_lines(d32_d3)
#plot_decision_boundary(Dbig_permutation, d32_d3, X1_lines, X2_lines)
#n.append(get_number_of_nodes(d32_d3))
#get_number_of_nodes(d32_d3)
##print("Printing n and errn lists respectively:")
##print(n)
##print(err)
##plt.plot(n,err)

