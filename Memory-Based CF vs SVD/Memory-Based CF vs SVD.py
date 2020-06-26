"""Dataset is from https://grouplens.org/datasets/movielens/1m/  """
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

print('Memory-Based CF by computing cosine similarity')
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
    

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
#print(item_prediction)

sparsity=round(1.0-len(df)/float(n_users*df.item_id.unique().shape[0]),3)
print ('The sparsity level of MovieLens1M is ' +  str(sparsity*100) + '%')

############################### SVD model based CF
print('Model-Based CF using Single Value Distance')
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print ('SVD model User-based CF RMSE: ' + str(rmse(X_pred, test_data_matrix)))