# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:19:31 2019

@author: Cuong Ngo
netid: cpn180001
CS6375 - Machine Learning. Spring 2019
Assignment 1
This program implements decision tree algorithm and post pruning algorithm
This program accepts 6 arguments
2 positive integers L and K  for the pruning algorithm
3 data set paths: training, test, and validationdata set
1 string input to trigger print, accept 'yes' or 'no'
"""
import sys
import pandas as pd
import numpy as np
import random
from copy import deepcopy
pd.set_option('display.max_columns', None)  


# function to validate positive integer 
def validate_positive_int(an_input):
    try:
        num = int(an_input)
        if num > 0:
            return True
        else:
            raise ValueError('Invalid input')
    except ValueError:
        print("Invalid input")
    
# taking 6 arguments and validate inputs
L_value = int(sys.argv[1]) if (validate_positive_int(sys.argv[1])) else -1
K_value = int(sys.argv[2]) if (validate_positive_int(sys.argv[2])) else -1

training_set_path = sys.argv[3]
validation_set_path = sys.argv[4]
test_set_path = sys.argv[5]
to_print = sys.argv[6]
needed_to_print = True if to_print == 'yes' else False


def import_training_set():    
    training_set = pd.read_csv(training_set_path, sep=',')
    return training_set

def import_test_set():    
    test_set = pd.read_csv(test_set_path, sep=',')
    return test_set

def import_validation_set():    
    validation_set = pd.read_csv(validation_set_path, sep=',')
    return validation_set

# tree data structure to hold the decision tree
class MyNode: 
    def __init__(self, name):
        self.name = name

# queue data structure to numerate the node in decision tree using bfs       
class MyQueue:
    def __init__(self):
        self.queue = []
        
    def enqueue(self, node):
        self.queue.insert(0,node)
    
    def dequeue(self):
        return self.queue.pop()
    
    def size(self):
        return len(self.queue) 

# custom log2 function to handle base 0 case    
def my_log2(number):
    if number != 0:
        return np.log2(number)
    else:
        return 0

# function to calculate entropy given p_plus and p_minus
def entropy(p_plus, p_minus):
    p_total = p_plus + p_minus
    if p_total == 0:
        return 0
    else:    
        return -p_plus/p_total * my_log2(p_plus/p_total) - p_minus/p_total * my_log2(p_minus/p_total)
  
# function to calculate varience impurity heuristic
def vi(k_0, k_1):
    k_total = k_0 + k_1
    if (k_total == 0):
        return 0
    else:
        return (k_0 * k_1)/(k_total * k_total)

# count number of rows in data that the attribute has value
# select * from data where attribute = value
def my_count(data, attribute, value):
    return data[data[attribute] == value].shape[0]    

# count number of row in data that has a specific value in attribute and specific value in target attribute
# select * from data where attribute = attribute_value and target_attribute = target_value
def my_count2(data, attribute, attribute_value, target_attribute, target_value):
    return data[(data[attribute] == attribute_value) & (data[target_attribute] == target_value)].shape[0]

    
# function to calculate the information gain using entropy
def info_gain1(examples, attribute):
    target_attribute = 'Class'
    # number of samples
    big_total = examples.shape[0]
    # entropy against target attribute
    big_entropy = entropy(my_count(examples, 'Class', 1),my_count(examples, 'Class', 0))
    # number of sample with attribute x = 1
    plus_total = my_count(examples, attribute, 1)
    # number of sample with attribute x = 0
    minus_total = my_count(examples, attribute, 0)
    # entropy of attribute x = 1
    plus_entropy = entropy(my_count2(examples, attribute, 1, target_attribute, 1), my_count2(examples, attribute, 1, target_attribute, 0))
    # entropy of attribute x = 0
    minus_entropy = entropy(my_count2(examples, attribute, 0, target_attribute, 1), my_count2(examples, attribute, 0, target_attribute, 0))
    # info gain calculation
    info_gain = big_entropy - (plus_total / big_total * plus_entropy) - (minus_total / big_total * minus_entropy)
    return attribute, info_gain

# function to calculate the information gain using variance impurity
def info_gain2(examples, attribute):
    target_attribute = 'Class'
    big_total = examples.shape[0]
    big_vi = vi(my_count(examples, 'Class', 1), my_count(examples, 'Class', 0))
    plus_total = my_count(examples, attribute, 1)
    minus_total = my_count(examples, attribute, 0)
    plus_vi = vi(my_count2(examples, attribute, 1, target_attribute, 1), my_count2(examples, attribute, 1, target_attribute, 0))
    minus_vi = vi(my_count2(examples, attribute, 0, target_attribute, 1), my_count2(examples, attribute, 0, target_attribute, 0))
    info_gain = big_vi - (plus_total / big_total * plus_vi) - (minus_total / big_total * minus_vi)
    
    return attribute, info_gain

# recursive id3 algorithm to build decision tree
def id3(examples, target_attribute, attributes, heuristic):
    
    root = MyNode('root')
    examples_size = examples.shape[0]
    positive_examples = examples[examples['Class'] == 1].shape[0]
    negative_examples = examples[examples['Class'] == 0].shape[0]
    
    
    if (examples_size == positive_examples):
        root.name = 1
        return root
    if (examples_size == negative_examples):
        root.name = 0
        return root
    
    most_common_attributes = 1 if positive_examples >= negative_examples else 0
    root.most_common_attributes = most_common_attributes
    if ( attributes.size == 0):
        
        root.name =  most_common_attributes
        return root
    
    next_attr = findBestAttributes(examples, attributes, heuristic)
    root.name = next_attr

    l_examples = examples.loc[examples[next_attr] != 1]
    if (l_examples.shape[0] == 0):
        root.lchild =  MyNode(most_common_attributes)
    else:
        root.lchild = id3(l_examples, target_attribute, attributes.drop(next_attr), heuristic)
        
    # handle right child - with attr and value = 1
    r_examples = examples.loc[examples[next_attr] != 0]    
    if (r_examples.shape[0] == 0):
        root.rchild = MyNode(most_common_attributes)
    else:
        root.rchild = id3(r_examples, target_attribute, attributes.drop(next_attr), heuristic)
    return root

# function to find the next best attribute (with maximum info gain) 
#   given the training set and a set of attribute
def findBestAttributes(examples, attributes, heuristic):
    bestAttr = ''
    maxInfoGain = 0
    for attr in attributes:
        # determine info_gain base on the type of heuristic
        attr_info_gain = info_gain1(examples, attr)[1] if (heuristic == 1) else info_gain2(examples, attr)[1]
        if (attr_info_gain >= maxInfoGain):
            bestAttr = attr
            maxInfoGain = attr_info_gain
    return bestAttr

def dfs_print(node, level):
    if (node.lchild.name == 0 or node.lchild.name == 1):
        print (" |  " * level, node.name, "= 0 :", node.lchild.name)
    else:
        print (" |  " * level, node.name, " = 0 : ")
        dfs_print(node.lchild, level+1)
    
    if (node.rchild.name == 0 or node.rchild.name == 1):
        print (" |  " * level, node.name, "= 1 :", node.rchild.name)
    else:
        print (" |  " * level, node.name, " = 1 : ")
        dfs_print(node.rchild, level+1)
    
    return level
    
def validate_test_set(root, test):
    correct = 0
    total = test.shape[0]
    
    tmpNode = root
    for index, row in test.iterrows():
        result = row["Class"]
        while (True):
            if(tmpNode.name == 0 or tmpNode.name == 1):
                if(result == tmpNode.name):
                    correct +=1 
                # reset to root for next line
                tmpNode = root
                break
            
            if (row[tmpNode.name] == 0):
                tmpNode = tmpNode.lchild
            else:
                tmpNode = tmpNode.rchild
    
    return correct/total   

# post prune algorithm
def post_prune(L_value, K_value, tree, validation_set):
    # set intial tree as the best tree with highest accuracy
    tree_best = deepcopy(tree)
    # get size of the tree
    size_initial = bfs_label_tree(tree)
    accuracy_initial = validate_test_set(tree_best, validation_set)
    accuracy_best = accuracy_initial
    
    # L value is for number of trees we want to try
    # K value is maximum of times we want to replace a subtree with a leaf
    # for each tree
    for i in range(1, L_value+1):
        tree_tmp = deepcopy(tree)
        m = random.randint(1, K_value+1)
        for j in range(1, m+1):  
            size_N = bfs_label_tree(tree_tmp)
            # pick a random node to be replace, max is the size of the tree
            p_value = random.randint(2, size_N+1)
            replace_subtree_with_leaf(tree_tmp, p_value)
        # check if the new tree has better accuracy, if yes, record it as
        # the best tree so far
        tmp_accuracy = validate_test_set(tree_tmp, validation_set)
        if (tmp_accuracy > accuracy_best):
            accuracy_best = tmp_accuracy
            tree_best = deepcopy(tree_tmp)
    # relabel the tree and get best tree's size before returning
    size_best = bfs_label_tree(tree_best)
    return accuracy_initial, tree_best, accuracy_best, size_initial, size_best

# function to level order tree, the level is stored in label attribute
def bfs_label_tree(root):
    my_queue = MyQueue()
    my_queue.enqueue(root)
    label = 1
    while(my_queue.size() != 0):
        node = my_queue.dequeue()
        
        # corner case if root  does not have children
        if(not hasattr(node, 'lchild')):
            node.label = label
            return label
            
        if( node.lchild.name not in [0,1]):
            my_queue.enqueue(node.lchild)
        if(node.rchild.name not in [0,1]):    
            my_queue.enqueue(node.rchild)
        node.label = label
        label += 1
    return label-1 

# function to replace a subtree at its node p th from root
def replace_subtree_with_leaf(root, p_value):
    # use queue to bfs until the node p, then replace it with
    # most common attribute of the subtree at p th node
    my_queue = MyQueue()
    my_queue.enqueue(root)
    while(my_queue.size() != 0):
        node = my_queue.dequeue()
        if (node.label == p_value):
            node.name = node.most_common_attributes
            delattr(node, 'lchild')
            delattr(node, 'rchild')
            break
        if( node.lchild.name not in [0,1]):
            my_queue.enqueue(node.lchild)
        if(node.rchild.name not in [0,1]):    
            my_queue.enqueue(node.rchild)    
    return
        
 # input and output logic   
training_set = import_training_set()
test_set = import_test_set()
validation_set = import_validation_set()

# heuristic 1 process
decision_tree_heuristic_1 = id3(training_set, 'Class', training_set.columns.drop('Class'),1)   
accuracy = validate_test_set(decision_tree_heuristic_1, test_set)
accuracy_initial, tree_best, accuracy_best, size_initial, size_best = post_prune(L_value, K_value, decision_tree_heuristic_1, validation_set)

f= open("outputs.txt","a+")
orig_stdout = sys.stdout
sys.stdout = f
print('Training set: ', training_set_path)
print('L=', L_value,' K=', K_value)
print('Test set accuracy - 1st heuristic = ', accuracy)
print('Original validation set accuracy - 1st heuristic =', accuracy_initial)
print('Post prune validation set accuracy - 1st heuristic =', accuracy_best)
# print the tree to report file
print('Initial decision tree - 1st heuristic (', size_initial, ' nodes)')
dfs_print(decision_tree_heuristic_1,0)
print('#################################################')
print('Post Pruning decision tree - 1st heuristic (', size_best, ' nodes)')
dfs_print(tree_best,0)
f.close()

#change out put back to standard output and check if the tree is needed to be printed in the console
sys.stdout = orig_stdout
print('Training set: ', training_set_path)
print('L=', L_value,' K=', K_value)
print('Test set accuracy - 1st heuristic = ', accuracy)
print('Original validation set accuracy - 1st heuristic =', accuracy_initial)
print('Post prune validation set accuracy - 1st heuristic =', accuracy_best)
if (needed_to_print):
    print('Initial decision tree - 1st heuristic (', size_initial, ' nodes)')
    dfs_print(decision_tree_heuristic_1,0)
    print('#################################################')
    print('Post Pruning decision tree - 1st heuristic (', size_best, ' nodes)')
    dfs_print(tree_best,0)

# heuristic 2 process
training_set = import_training_set()
decision_tree_heuristic_2 = id3(training_set, 'Class', training_set.columns.drop('Class'),2)
accuracy = validate_test_set(decision_tree_heuristic_2, test_set)
accuracy_initial, tree_best, accuracy_best, size_initial, size_best = post_prune(L_value, K_value, decision_tree_heuristic_2, validation_set)

f= open("outputs.txt","a")
orig_stdout = sys.stdout
sys.stdout = f
print('#################################################') 
print('Test set accuracy - 2nd heuristic = ', accuracy)
print('Original validation set accuracy - 2nd heuristic =', accuracy_initial)
print('Post prune validation set accuracy - 2nd heuristic =', accuracy_best)

# print the trees to output file
print('Initial decision tree - 2nd heuristic(', size_initial, ' nodes)')
dfs_print(decision_tree_heuristic_2,0)
print('#################################################')
print('Post Pruning decision tree - 2nd heuristic(', size_best, ' nodes)')
dfs_print(tree_best,0)
f.close()

sys.stdout = orig_stdout
print('#################################################') 
print('Test set accuracy - 2nd heuristic = ', accuracy)
print('Original validation set accuracy - 2nd heuristic =', accuracy_initial)
print('Post prune validation set accuracy - 2nd heuristic =', accuracy_best)
if (needed_to_print):
    print('Initial decision tree - 2nd heuristic(', size_initial, ' nodes)')
    dfs_print(decision_tree_heuristic_2,0)
    print('#################################################')
    print('Post Pruning decision tree - 2nd heuristic(', size_best, ' nodes)')
    dfs_print(tree_best,0)
print ('============================== End of a command ===============================')


   

