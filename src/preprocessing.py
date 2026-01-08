# only add Preprocessors for data to this file

# Imports

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans

#Basic Transfomers

#Imputer
imputer = SimpleImputer(strategy="median")
# imputer.set_output(transform="pandas")

#Ordinal Encoder
ordinal_encoder = OrdinalEncoder()

#One Hot Encoder cat = category
cat_encoder = OneHotEncoder(sparse_output=False)

#Min max Scaler
min_max_scaler = MinMaxScaler(feature_range=(-1,1))

#Standard Scaler
std_scaler = StandardScaler()

#Rbf Kernel (Similarity)
#import and change Y accordingly
rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))
# LOG transformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)

#Ratio Transformer
def column_ratio(X):
    """
    Calculates the ratio between the first and second column.
    Input X: A 2D array with 2 columns.
    Returns: A 2D array with 1 column (Column 0 / Column 1).
    """
    # We use X[:, [0]] (with brackets) to keep it 2D.
    # If we used X[:, 0], it would become a flat 1D list, which Scikit-Learn dislikes.
    return X[:, [0]] / X[:, [1]]
ratio_transformer = FunctionTransformer(column_ratio)

#Standard scaler Clone

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

# Cluster similarity
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!
# after training the cluster centers are available with cluster center attribute.
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]