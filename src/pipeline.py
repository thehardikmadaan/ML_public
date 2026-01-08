# only Add Pipeline Transformers to this file

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#First impute than Scale the input

#num_pipeline = Pipeline([("impute", SimpleImputer(strategy="meidan")]),("standardize", StandardScaler())]
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

#Category pipeline , converts text to numerical 1hot standard
cat_pipeline = make_pipeline(
    SimpleImputer(strategy= "most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)


