import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('f://VIT Codes/CF_100k/u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]

n_items = df.item_id.unique().shape[0]
print( 'Number of users =',str(n_users))
print('  Number of movies = ',str(n_items)  )

#calculating sparsity level
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print("The Sparsity level of Movielens 100k is",str(sparsity*100),"%")

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)
print("Training data is")
print(train_data)
type(train_data)
#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
                train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
                test_data_matrix[line[1]-1, line[2]-1] = line[3]


import scipy.sparse as sp
from scipy import linalg 
from sklearn.decomposition import PCA


pca=PCA(n_components=2)
pca.fit(train_data_matrix)

s_diag_matrix=np.diag(s)
X_pred=np.dot(np.dot(u,s_diag_matrix),vt)

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
                prediction = prediction[ground_truth.nonzero()].flatten() 
                ground_truth = ground_truth[ground_truth.nonzero()].flatten()
                return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ',str(rmse(X_pred, test_data_matrix)))
