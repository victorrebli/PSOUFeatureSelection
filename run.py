from importlib import reload
from scipy.io import arff
#import utils
#reload(utils)
#from ETree import Tree_expression, ParseTree
import copy
#import evaluate
#from evaluate import GPFactory
#reload(evaluate)
import principal
from principal import PSOSelector
reload(principal)
import lightgbm as lgb
import numpy as np
import collections
import pandas as pd
import sys
from sklearn.cluster import KMeans

data = arff.loadarff('bases/sonar.arff')
df_data = pd.DataFrame(data[0])

y_train = df_data.Class
X_train = df_data.drop('Class', axis=1)
y_train = np.where(y_train == b'Rock', 1, 0)

estimator = KMeans(n_clusters=2, random_state=0)

extras = {'eval_metric': 'auc',
          'early_stopping_rounds': 100,
         'verbose': False}

model = PSOSelector(estimator, w=0.7298, c1=1.49618, c2=1.49618,
                    num_particles=30, max_iter=100, max_local_improvement=50,
                    maximize_objective=True, initialization='uniform',
                    fitness_method='type_2', cv = 3)

print(X_train.shape)
model.fit(X_train)                             