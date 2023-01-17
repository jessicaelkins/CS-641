# Name: Jessica Elkins
# Date: 9/22/2022
# Class: UAH CS 641 FA 2022
# This program implements a decision tree to classify 
# cars into the categories of "Unacceptable", "Acceptable",
# "Good", or "Very Good" based on the given dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

car_data = pd.read_csv('car.data', header = None)
car_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# preprocessing / encoding the dataset
buying_n = LabelEncoder() 
maint_n = LabelEncoder()
doors_n = LabelEncoder()
persons_n = LabelEncoder()
boot_n = LabelEncoder()
safety_n = LabelEncoder()
class_n = LabelEncoder()

car_data['buying_n'] = buying_n.fit_transform(car_data['buying'])
car_data['maint_n'] = maint_n.fit_transform(car_data['maint'])
car_data['doors_n'] = doors_n.fit_transform(car_data['doors'])
car_data['persons_n'] = persons_n.fit_transform(car_data['persons'])
car_data['boot_n'] = boot_n.fit_transform(car_data['lug_boot'])
car_data['safety_n'] = safety_n.fit_transform(car_data['safety'])
car_data['class_n'] = class_n.fit_transform(car_data['class'])

F = ['buying_n', 'maint_n', 'boot_n', 'safety_n', 'doors_n', 'persons_n', 'class_n']
E = car_data[F]

def seperate_train_test(E):
    objects_num = len(E)
    # 80% training, 20% test
    train_num = round(0.8 * objects_num)
    test_num = round(0.2 * objects_num)
    
    # empty list to store random numbers generated
    random_numbers = []
    
    # initialize traning df and testing df
    train = pd.DataFrame(columns = F)
    test = pd.DataFrame(columns = F)
    
    for i in range(train_num):
        while True:
            # generate random number within the range of dataset
            random_num = random.randint(0, objects_num-1)
            # make sure the number has not been used already
            if random_num not in random_numbers:
                break
        # add that number to the list of numbers generated
        random_numbers.append(random_num)
        # add that point to the training set
        train.loc[len(train.index)] = E.iloc[random_num]
        
    for i in range(test_num):
        while True:
            random_num = random.randint(0, objects_num-1)
            if random_num not in random_numbers:
                break
        random_numbers.append(random_num)
        test.loc[len(test.index)] = E.iloc[random_num]
    
    return train, test

train, test = seperate_train_test(E)
train_x = train.drop(['class_n'], axis = 'columns')
train_y = train.iloc[:, -1]
test_x = test.drop(['class_n'], axis = 'columns')
test_y = test.iloc[:, -1]

# node class that the TreeGrowth function uses
class createNode:
    def __init__(self, test_cond = None, label = None, left_subtree = None, right_subtree = None):
        self.test_cond = test_cond
        self.label = label
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree

# given a dataframe, calculate the gini
def calculate_gini(df):
    acc, good, unacc, vgood = 0, 0, 0, 0
    total = len(df)
    
    for x in df:
        if x == 0:
            acc += 1
        elif x == 1:
            good += 1
        elif x == 2:
            unacc += 1
        elif x == 3:
            vgood += 1
    
    p_acc = acc / total
    p_good = good / total
    p_unacc = unacc / total
    p_vgood = vgood / total
    
    gini = 1 - (p_acc**2 + p_good**2 + p_unacc**2 + p_vgood**2)
    return gini

def stopping_cond(E, F):
    # check if the gini is 0 or 
    # minimum number of samples is violated
    min_samples = 2
    if len(E) <= min_samples:
        return True
    
    class_row = E.iloc[:, -1]
    gini = calculate_gini(class_row)
    
    if gini == 0:
        return True
    return False
        
def find_best_split(E, F):
    # go through each split and find the best one using gini.
    # will return a dictionary 
    test_cond = {}
    best_gini = 1
    
    for i in range(len(F) - 1):
        # for each attribute in E, create an array of unique attribute
        # values so that we can calculate the gini and gain from trying
        # each unique attribute value as a split 
        attribute_vals = E.iloc[:, i]
        test_splits = np.unique(attribute_vals)
        
        for split in test_splits:
            # convert df to numpy array to easily split into left subtree and right subtree
            E_arr = E.to_numpy()
            left_subtree_arr = np.array([row for row in E_arr if row[i] < int(split)])
            right_subtree_arr = np.array([row for row in E_arr if row[i] >= int(split)])
            
            # make sure the array is not empty before converting to df
            if ((len(left_subtree_arr) != 0) and (len(right_subtree_arr) != 0)):
                left_subtree = pd.DataFrame(left_subtree_arr, columns = F)
                right_subtree = pd.DataFrame(right_subtree_arr, columns = F)
            
                # calculate gini of root node
                root_class = E.iloc[:, -1]
                root_gini = calculate_gini(root_class)
                
                # calculate gini of left subtree node
                left_class = left_subtree.iloc[:, -1]
                left_gini = calculate_gini(left_class)
                
                # calculate gini of right subtree node
                right_class = right_subtree.iloc[:, -1]
                right_gini = calculate_gini(right_class)
            
                # calculate total gini
                weight1 = len(left_class) / len(root_class)
                weight2 = len(right_class) / len(root_class)
                weighted_gini = (weight1 * left_gini) + (weight2 * right_gini)
                
                # if the gini of the current split is lower than lowest gini found so far,
                # update with new split
                if weighted_gini < best_gini:
                    test_cond['left_subtree'] = left_subtree
                    test_cond['right_subtree'] = right_subtree
                    test_cond['attribute_i'] = i
                    test_cond['split'] = split
                    best_gini = weighted_gini
                    
    return test_cond

# look at the class of each datapoint and assign node label based on majority
# acc -> 0, good -> 1, unacc -> 2, vgood -> 3
def Classify(E):
    class_row = E.iloc[:, -1]
    acc, good, unacc, vgood = 0, 0, 0, 0
    total = len(class_row)
    
    for datapoint in class_row:
        if datapoint == 0:
            acc += 1
        elif datapoint == 1:
            good += 1
        elif datapoint == 2:
            unacc += 1
        elif datapoint == 3:
            vgood += 1
        
    p_acc = acc / total
    p_good = good / total
    p_unacc = unacc / total
    p_vgood = vgood / total
    
    p_dict = {0 : p_acc, 1 : p_good, 2 : p_unacc, 3 : p_vgood}
    argmax, label = 0, None
    
    for x in p_dict:
        if p_dict[x] > argmax:
            argmax = p_dict[x]
            label = x
    
    return label
    
# build the decision tree
# modeled after p. 137 algorithm
def TreeGrowth(E, F):
    if stopping_cond(E, F):
        leaf = createNode()
        leaf.label = Classify(E)
        return leaf
    else:
        root = createNode()
        root.test_cond = find_best_split(E, F)
        # recursively build left subtree
        root.left_subtree = TreeGrowth(root.test_cond['left_subtree'], F)
        # recursively build right subtree
        root.right_subtree = TreeGrowth(root.test_cond['right_subtree'], F)
    return root

root = TreeGrowth(train, F)

# given a data point and root/node from tree
# return the label of the data point
def classify_object(point, node):
    if node.label is not None:
        return node.label
    else:
        attribute_i = node.test_cond['attribute_i']
        split = node.test_cond['split']
        if point[attribute_i] < split:
            return classify_object(point, node.left_subtree)
        else:
            return classify_object(point, node.right_subtree)
            
# given dataset and root from tree
# return an array with class labels
def classify_dataset(E, root):
    length = len(E)
    label_array = []
    
    for i in range(length):
        label = classify_object(E.iloc[i], root)
        label_array.append(label)
        
    return label_array

# classifying the test data
classify_result = classify_dataset(test_x, root)

# classifying the training data
classify_train = classify_dataset(train_x, root)

def calculate_accuracy(classify_result, answers):
    total = len(classify_result)
    total_correct = 0
    
    for i in range(total):
        if classify_result[i] == answers[i]:
            total_correct += 1
    
    return total_correct / total

# calculating testing accuracy and error rate
accuracy_test = calculate_accuracy(classify_result, test_y)
print('Accuracy on test data: ', accuracy_test)
print('Error rate during testing: ', 1 - accuracy_test)

# calculating training accuracy and error rate
accuracy_train = calculate_accuracy(classify_train, train_y)
print('Accuracy on training data: ', accuracy_train)
print('Error rate during training: ', 1 - accuracy_train)

# recursive function to count number of nodes given a tree
def num_of_nodes(node):
    if node is None:
        return 0
    else:
        return num_of_nodes(node.left_subtree) + num_of_nodes(node.right_subtree) + 1

# recursive function to count number of leaf nodes given a tree   
def num_of_leaf_nodes(node):
    if node is None:
        return 0
    if(node.left_subtree is None and node.right_subtree is None):
        return 1
    else:
        return num_of_leaf_nodes(node.left_subtree) + num_of_leaf_nodes(node.right_subtree)

count = num_of_nodes(root)
print('Number of nodes: ', count)

count2 = num_of_leaf_nodes(root)
print('Number of leaf nodes: ', count2)