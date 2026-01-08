from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

#First impute than Scale the input

#num_pipeline = Pipeline([("impute", SimpleImputer(strategy="meidan")]),("standardize", StandardScaler())]
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

