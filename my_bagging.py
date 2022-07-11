from random import seed
from random import random
from random import randrange
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
import math
import copy

# Merger
def merger(training_samples, training_labels):
    labeled_samples = []
    for sample, label in zip(training_samples, training_labels):
        l_sample = np.append(sample, int(label))
        labeled_samples.append(l_sample)
        
    return labeled_samples


# Create a random subsample from the dataset with replacement. ND Array
def bts(X):
    nsample = 1
    tmp = []
    for i in range(nsample):
        id_pick = np.random.choice(np.shape(X)[0], size = (np.shape(X)[0]))
        boot1 = X[id_pick,:]
        tmp =  boot1

    return tmp

def bagging(dataset, labels, model = 'KNN', problem_type = 'classification', B = 50):
    
    predictions = []
    labeled_dataset = np.array(merger(dataset, labels))
    OOB_bootstrapped_bool_list = []
    boostrapped_sets = []
    prediction_rules = []
    for i in range(B):
        new_bts = bts(labeled_dataset)

        check = [True] * len(new_bts)
        OOB_preds = []
        
        learners = {'DecisionTree':[tree.DecisionTreeClassifier(random_state =  0, max_depth = 2),
                                tree.DecisionTreeRegressor(random_state = 0, max_depth = 2)],
                
               'KNN':[neighbors.KNeighborsClassifier(n_neighbors = 1),neighbors.KNeighborsRegressor(n_neighbors = 1)],
                
                'SVM':[svm.SVC(), svm.SVR()],
                
                'Ridge':[linear_model.RidgeClassifier(), linear_model.Ridge(alpha = 1.0)]
                }
        new_model = learners[model][0] if problem_type == 'classification' else learners[model][1]
        
            
        X = new_bts[:, :-1]
        Y = new_bts[:, -1]
        prediction_rules.append(new_model.fit(X,Y))
        
        for i in range(len(labeled_dataset)):
            
            temp = (labeled_dataset[:, :-1])[i].reshape(1, -1)

            OOB_preds.append(prediction_rules[-1].predict(temp))

            for j in range(len(new_bts)):
                if (labeled_dataset[i] == new_bts[j]).all():
                    check[i] = False

        boostrapped_sets.append(np.array(merger(new_bts, OOB_preds)))
        OOB_bootstrapped_bool_list.append(check)
  

    new_bagging_output = BaggingOutput()
    new_bagging_output.original_dataset = labeled_dataset
    new_bagging_output.problem_type = problem_type
    new_bagging_output.model = model
    new_bagging_output.boostrapped_sets = np.array(boostrapped_sets)
    new_bagging_output.OOB_bootstrapped_bool_list =np.array(OOB_bootstrapped_bool_list)
    new_bagging_output.prediction_rules = prediction_rules

    return new_bagging_output
    
