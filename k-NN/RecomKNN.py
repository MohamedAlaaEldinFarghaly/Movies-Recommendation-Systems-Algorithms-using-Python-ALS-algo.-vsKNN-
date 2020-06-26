
""" Memory-Based Recommender System using KNN

Dataset is from the folder https://grouplens.org/datasets/movielens/1m/
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv


import random
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


import matplotlib.pyplot as plt

# Create User-Item Matrix

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ratings.dat', sep='::', names=header,engine='python')
#print (df.item_id)
n_users = df.user_id.unique().shape[0]
#n_items = df.item_id.unique().shape[0]
n_items = max(df.item_id)

print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
train_data, test_data = cv.train_test_split(df, test_size=0.1)

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
#print (train_data.itertuples())
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

W = train_data_matrix>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)    

Y =test_data_matrix>0.5
Y[Y == True] = 1
Y[Y == False] = 0
# To be consistent with our Q matrix
Y = Y.astype(np.float64, copy=False)   
    
# Cosine metric implemintation
metric ='cosine'
ks=list(range(5,200,5))
training=W
testing=Y
print ("Training and scoring")
scores = []
knn = NearestNeighbors(metric=metric, algorithm="brute")
knn.fit(training)
aucmat=[]
for k in ks:
    print ("Evaluating for", k, "neighbors")
    neighbor_indices = knn.kneighbors(testing,
                                      n_neighbors=k,
                                      return_distance=False)

    all_predicted_scores = []
    all_labels = []
    notsampledpredicted= np.zeros((n_users, n_items))
    for user_id in range(testing.shape[0]):
        user_row = testing[user_id, :]
        nonZ=user_row.nonzero()
        #Labeling
        interaction_indices= list(nonZ[0])
        interacted = set(interaction_indices)
        non_interacted = set(range(testing.shape[1])) - interacted

        n_samples = min(len(non_interacted), len(interacted))
        sampled_interacted = random.sample(interacted, n_samples)
        sampled_non_interacted = random.sample(non_interacted, n_samples)

        indices = list(sampled_interacted)
        indices.extend(sampled_non_interacted)
        labels = [1] * n_samples
        labels.extend([0] * n_samples)
        
        neighbors = training[neighbor_indices[user_id, :], :]
        predicted_scores = neighbors.mean(axis=0)
        notsampledpredicted[user_id,:]=predicted_scores
        for idx in indices:
            all_predicted_scores.append(predicted_scores[idx])
        all_labels.extend(labels)

    print (len(all_labels), len(all_predicted_scores))

    auc = roc_auc_score(all_labels, all_predicted_scores)

    print ("k", k, "AUC", auc)
    aucmat=np.append(aucmat,auc)
    

##Euclidean metric KNN
metric ='euclidean'
ks=list(range(5,200,5))
training=W
testing=Y
print ("Training and scoring")
scores = []
knn = NearestNeighbors(metric=metric, algorithm="brute")
knn.fit(training)
aucmat2=[]
for k in ks:
    print ("Evaluating for", k, "neighbors")
    neighbor_indices = knn.kneighbors(testing,
                                      n_neighbors=k,
                                      return_distance=False)

    all_predicted_scores = []
    all_labels = []
    notsampledpredicted= np.zeros((n_users, n_items))
    for user_id in range(testing.shape[0]):
        user_row = testing[user_id, :]
        nonZ=user_row.nonzero()
        
        interaction_indices= list(nonZ[0])
        interacted = set(interaction_indices)
        non_interacted = set(range(testing.shape[1])) - interacted

        n_samples = min(len(non_interacted), len(interacted))
        sampled_interacted = random.sample(interacted, n_samples)
        sampled_non_interacted = random.sample(non_interacted, n_samples)

        indices = list(sampled_interacted)
        indices.extend(sampled_non_interacted)
        labels = [1] * n_samples
        labels.extend([0] * n_samples)
        
        neighbors = training[neighbor_indices[user_id, :], :]
        predicted_scores = neighbors.mean(axis=0)
        for idx in indices:
            all_predicted_scores.append(predicted_scores[idx])
        all_labels.extend(labels)

    print (len(all_labels), len(all_predicted_scores))

    auc = roc_auc_score(all_labels, all_predicted_scores)

    print ("k", k, "AUC", auc)
    aucmat2=np.append(aucmat2,auc)
    
# Plotting
plt.plot(ks,aucmat)
plt.plot(ks, aucmat2)
plt.legend(['cosine', 'euclidean'])
plt.xlabel('K')
plt.ylabel('Area Under Curve')