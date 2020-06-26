#Please use the data set in the same folder
#Also change the path in the codes below to read the data set properly into tables
#Also note that the code takes a long time to output results (About an hour for me)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#This part is for reading the dababase tables
tag_headers = ['user_id', 'movie_id', 'tag', 'timestamp']
tags = pd.read_csv('E:/Studying/Year4/Machine learning/CF project/ml-latest-small/tags.csv', sep=',', header=None, names=tag_headers)

rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('E:/Studying/Year4/Machine learning/CF project/ml-latest-small/ratings.csv', sep=',', header=None, names=rating_headers)

movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_csv('E:/Studying/Year4/Machine learning/CF project/ml-latest-small/movies.csv',sep=',', header=None, names=movie_headers)
movie_titles = movies.title.tolist()


ratings_test=pd.read_csv('E:/Studying/Year4/Machine learning/CF project/ml-latest-small/test.csv', sep=',', header=None, names=rating_headers)
###############

#This part is for joining the tables together
l = pd.merge(movies,ratings,on=['movie_id'],how='outer')
df = pd.merge(l,tags,on=['movie_id'],how='outer')
del df['timestamp_y']
del df['user_id_y']

l2 = pd.merge(movies,ratings_test,on=['movie_id'],how='outer')
df2 = pd.merge(l2,tags,on=['movie_id'],how='outer')
del df2['timestamp_y']
del df2['user_id_y']


####################

#This part is for creating the user/rating table from the previous tables
table = df.pivot_table( index=['user_id_x'],columns=['movie_id'],values='rating',aggfunc='first',fill_value =0)
table_test = df2.pivot_table( index=['user_id_x'],columns=['movie_id'],values='rating',aggfunc='first',fill_value =0)
Q=table.values
Q_test = table_test.values





#############
W = Q>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)
lambda_ = 0.1
#Number of features is n_factors
n_factors = 100
m, n = Q.shape
n_iterations = 20
X = 5 * np.random.rand(m, n_factors) #initialize random values for X and Y
Y = 5 * np.random.rand(n_factors, n)
#Calculate the error
def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)


#########
#Iterate on the matrix X and Y until convergence

weighted_errors = []
for ii in range(4):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    weighted_errors.append(get_error(Q, X, Y, W))
    print('{}th iteration is completed'.format(ii))
weighted_Q_hat = np.dot(X,Y)
##########

def print_recommendations(W=W, Q=Q, Q_hat=weighted_Q_hat, movie_titles=movie_titles):
    # Making Q range from 0 to 5
    Q_hat -= np.min(Q_hat)
    Q_hat *= float(5) / np.max(Q_hat)
    movie_ids = np.argmax(Q_hat - 5 * W, axis=1)
    #Print movies for users that have rating > 3
    for jj, movie_id in zip(range(m), movie_ids):
        #if Q_hat[jj, movie_id] < 0.1: continue
        print('User {} liked {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
        #print('User {} did not like {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
        print('\n User {} recommended movie is {} - with predicted rating: {}'.format(
                    jj + 1, movie_titles[movie_id], Q_hat[jj, movie_id]))
        print('\n' + 100 *  '-' + '\n')
#print_recommendations()

print_recommendations(Q_hat=weighted_Q_hat)
###################################
#testing
ind_test=np.nonzero(Q_test)  #get indices for non zero entries to compare it with the predicted ratings
NN,MM = np.shape(ind_test)
RMSE=0
#loop over the test values and predicted values and compare them using RMSE
for indexx in range(MM):
    a=ind_test[0][indexx] 
    b=ind_test[1][indexx]
    RMSE = RMSE + (weighted_Q_hat[a][b] - Q_test[a][b])**2
RMSE = RMSE / MM

print('Test RMSE = ', RMSE)

