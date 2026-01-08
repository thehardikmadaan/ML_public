# only Add Pipeline Transformers to this file
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

#First impute than Scale the input
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

#Category pipeline , converts text to numerical 1hot standard
cat_pipeline = make_pipeline(
    SimpleImputer(strategy= "most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
