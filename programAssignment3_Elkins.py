# Name: Jessica Elkins
# Date: 11/17/2022
# Class: UAH CS 641 FA 2022
# This program implements the K-Means clustering algorithm
# on the iris data set to classify iris types as "Iris Setosa",
# "Iris Versicolour", or "Iris Virginica".

import numpy as np
import pandas as pd

# importing the iris data set
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
column_names2 = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_data = pd.read_csv(r'iris.data', names=column_names)
iris_data['class'] = iris_data['class'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# separating the data by attributes and label
X = iris_data.iloc[:,0:4]
Y = iris_data.iloc[:,-1]

# number of clusters
K = 3

def initial_centroids(iris_data, K):
    centroids = []
    
    # separating the data by their classes
    class_0 = iris_data.iloc[0:50,:]
    class_1 = iris_data.iloc[50:100,:]
    class_2 = iris_data.iloc[100:150,:]
    
    # picking a centroid from each class
    centroids.append(class_0.sample(1))
    centroids.append(class_1.sample(1))
    centroids.append(class_2.sample(1))
    return centroids

def euclidean_distance(data_point, centroid):
    # separating the sepal len, sepal wid, petal len, petal wid for point and centroid
    x1, y1, z1, w1 = data_point[1], data_point[2], data_point[3], data_point[4]
    x2, y2, z2, w2 = centroid.iloc[:,0], centroid.iloc[:,1], centroid.iloc[:,2], centroid.iloc[:,3]
    
    # calculating the euclidean distance
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 + (w2 - w1)**2)
    return distance

def assign_points(iris_data, centroids):
    classes = []
    
    # loop through each data point and calculate its distance from each
    # of the 3 centroids. Assign the point to the closest centroid.
    for data_point in iris_data.itertuples():
        distance_to_c0 = euclidean_distance(data_point, centroids[0])
        distance_to_c1 = euclidean_distance(data_point, centroids[1])
        distance_to_c2 = euclidean_distance(data_point, centroids[2])
        distances = [distance_to_c0, distance_to_c1, distance_to_c2]
        classes.append(np.argmin(distances))   
    return classes

def update_centroids(iris_data, K, classes):
    new_centroids = []
    
    # for each centroid, take the mean of the points that belong to that centroid
    # and use that value as the new centroid
    for i in range(K):
        # initialize variables
        N, x_sum, y_sum, z_sum, w_sum = 0, 0, 0, 0, 0
        
        for data_point in iris_data.itertuples():
            j = data_point[0]
            
            #checking if the point belongs to the current cluster
            if i == classes[j]:
                N = N + 1
                x_sum += data_point[1]
                y_sum += data_point[2]
                z_sum += data_point[3]
                w_sum += data_point[4]
                
        # calculate the mean for each attribute
        x_mean = x_sum / N
        y_mean = y_sum / N
        z_mean = z_sum / N
        w_mean = w_sum / N
        
        # creating the new centroid
        new_centroid = pd.DataFrame([[x_mean, y_mean, z_mean, w_mean]], columns=column_names2)
        
        # adding it to the list of centroids
        new_centroids.append(new_centroid)
    
    # return the list of centroids
    return new_centroids

def calculate_sse(iris_data, classes, centroids):
    errors = []
    
    for data_point in iris_data.itertuples():
        # figuring out which centroid the data point belonged to
        index = data_point[0]
        centroid_number = classes[index]
        centroid = centroids[centroid_number]
        
        # separating the sepal len, sepal wid, petal len, petal wid for point and centroid 
        x1, y1, z1, w1 = data_point[1], data_point[2], data_point[3], data_point[4]
        x2, y2, z2, w2 = centroid.iloc[:,0], centroid.iloc[:,1], centroid.iloc[:,2], centroid.iloc[:,3]
        
        #calculate the error by squaring the distances of the point to its nearest centroid
        error = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 + (w2 - w1)**2
        errors.append(error)
    
    # sum the errors to get SSE
    sse = sum(errors)
    sse.to_numpy() # it was printing like a dataframe, so I converted to numpy array
    return sse[0]

def calculate_accuracy(classes, Y):
    total = len(classes)
    total_correct = 0
    
    for i in range(total):
        if classes[i] == Y[i]:
            total_correct += 1
    
    return total_correct / total

def kmeans(iris_data, K):
    # Select K = 3 points from each class as initial centroids
    centroids = initial_centroids(iris_data, K)
    previous_centroids, classes = [], []
    
    # Stop if the centroids do not change after updating
    while(not np.array_equal(previous_centroids, centroids)):
        # Assigning the points to their closest centroids
        classes = assign_points(iris_data, centroids)
    
        # Updating centroids
        previous_centroids = centroids
        centroids = update_centroids(iris_data, K, classes)
    
    # Calculate and print the SSE
    sse = calculate_sse(iris_data, classes, centroids)
    print(f"SSE: {sse:>4f}")
    
    return classes

classes = kmeans(iris_data, K) 
accuracy = calculate_accuracy(classes, Y)
print(f"Accuracy: {accuracy:>4f}")