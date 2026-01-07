import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import FunctionTransformer

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
#rbf_kernel = rbf_kernel()

# LOG transformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)

