import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('f:/VIT Codes/CF_1M/ratings_1m.csv', sep=',',names=header,dtype='int')
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
max_no_of_user_id=df['user_id'].max()
max_no_of_item_id=df['item_id'].max()
print( 'Number of users =',str(n_users))
print('  Number of movies = ',str(n_items)  )
print('Max user_id:',str(max_no_of_user_id))
print('Max item_id:',str(max_no_of_item_id))

#calculating sparsity level
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print("The Sparsity level of Movielens 100k is",str(sparsity*100),"%")

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)
print("Training data is")
print(train_data)
#type(train_data)
#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((max_no_of_user_id, max_no_of_item_id))
for line in train_data.itertuples():
                train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((max_no_of_user_id, max_no_of_item_id))
for line in test_data.itertuples():
                test_data_matrix[line[1]-1, line[2]-1] = line[3]



from sklearn.decomposition import TruncatedSVD

#svd=TruncatedSVD(n_components=5,n_iter=7,random_state=42)
#svd.fit(train_data_matrix)
#import scipy.sparse as sp
#from scipy.sparse.linalg import svds


u,s,vt=TruncatedSVD(train_data_matrix,k=20)
s_diag_matrix=np.diag(s)
X_pred=np.dot(np.dot(u,s_diag_matrix),vt)

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
                prediction = prediction[ground_truth.nonzero()].flatten() 
                ground_truth = ground_truth[ground_truth.nonzero()].flatten()
                return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE:',str(rmse(X_pred, test_data_matrix)))
