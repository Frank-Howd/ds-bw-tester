import pandas as pd
from joblib import load, dump

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # cv=5, random_state=22
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

classifier = load('../assets/lr_model.joblib')

print(classifier)