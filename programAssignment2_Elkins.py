# Name: Jessica Elkins
# Date: 10/20/2022
# Class: UAH CS 641 FA 2022
# This program implements a Multilayer Neural Network to classify 
# cars into the categories of "Unacceptable", "Acceptable",
# "Good", or "Very Good" based on the given dataset.

import torch
import random
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# importing the data as a pandas dataframe
car_data = pd.read_csv(r'car.data', header = None)
car_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_data.head()

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
car_data

# creating a dataframe with just the encoded columns
column_names = ['buying_n', 'maint_n', 'boot_n', 'safety_n', 'doors_n', 'persons_n', 'class_n']
E = car_data[column_names]

# A function that takes a pandas dataframe as input and 
# samples it into an 80/20 split without replacement.
# Returns two pandas dataframes
def seperate_train_test(E):
    objects_num = len(E)
    # 80% training, 20% test
    train_num = round(0.8 * objects_num)
    test_num = round(0.2 * objects_num)
    
    # empty list to store random numbers generated
    random_numbers = []
    
    # initialize traning df and testing df
    train = pd.DataFrame(columns = column_names)
    test = pd.DataFrame(columns = column_names)
    
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

# calling the seperate_train_test function to 80/20 split the data
train, test = seperate_train_test(E)

# seperating the class label column from the train and test data
train_x = train.drop(['class_n'], axis = 'columns')
train_y = train.iloc[:, -1]
test_x = test.drop(['class_n'], axis = 'columns')
test_y = test.iloc[:, -1]

# checking if a GPU is available for computation
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# converting the pandas dataframes to torch Float tensors
train_tensor = torch.FloatTensor(train.values)
x_train = torch.FloatTensor(train_x.values)
y_train = torch.FloatTensor(train_y.values)
x_test = torch.FloatTensor(test_x.values)
y_test = torch.FloatTensor(test_y.values)

# class that creates the neural network
class NeuralNetwork(nn.Module):
    # constructor that initializes the network layers and neurons
    def __init__(self, input_dimension, output_dimemsion):
        super(NeuralNetwork, self).__init__()
        
        self.l1 = nn.Linear(input_dim, 400)
        self.l2 = nn.Linear(400,300)
        self.l3 = nn.Linear(300, output_dim)

    # The forward propagation method that runs the data through the
    # network and applies the activation function.   
    def forward(self, input):
        q = F.relu(self.l1(input))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

# initializing the hyperparameters
input_dim = 6   # number of input dimensions
output_dim = 4  # number of output dimensions
batch_size = 64 # mini-batch size for SGD
learning_rate = 0.01 

# initializing the model
net = NeuralNetwork(input_dim, output_dim).to(device)

# choosing the loss function and optimizer 
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)

# converting the training data to a dataloader
train_dataloader = DataLoader(train_tensor, batch_size=64, shuffle=True)

# training the model using minibatches of 64 and SGD
def train_loop(dataloader, network, loss_function, optimizer):
    loss = 0
    for i, training_set in enumerate(dataloader):
        y_training = training_set[:,-1]
        x_training = training_set[:,:-1]
        
        # forward propagation to compute predictions
        predictions = net(x_training.to(device))
        y_training = y_training.type(torch.LongTensor).to(device)
        
        # compute loss
        loss = loss_function(predictions, y_training)
        
        # backwards propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        
    return loss

# a function that runs the test data through the model and outputs accuracy
def test_func(testing_x, testing_y, network):
    predictions = net(testing_x)
    correct = 0
    # sums up the amount of predictions that are correct
    correct = correct + (predictions.argmax(1) == testing_y).type(torch.float).sum().item()
    accuracy = correct / len(testing_y)
    return accuracy

epochs = 1000
last_accuracy = 0.00

for i in range(epochs):
    loss = train_loop(train_dataloader, net, loss_function, optimizer)
    accuracy = test_func(x_test, y_test, net)

    if i % 100 == 0 or i == epochs - 1:
        if last_accuracy == accuracy:
            print("Stopped due to possible overfitting")
            print("Epoch:",i)
            print(f" Loss: {loss:>5f}")
            print(f" Test Accuracy: {accuracy:>4f}")
        else:
            print("Epoch:",i)
            print(f" Loss: {loss:>5f}")
            print(f" Test Accuracy: {accuracy:>4f}")
            last_accuracy = accuracy
