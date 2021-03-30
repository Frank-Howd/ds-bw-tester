import pandas as pd
from joblib import load, dump

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # cv=5, random_state=22
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

df = pd.read_csv("assets/ks-projects-201801.csv")

filter = ( df['state'] == 'failed' ) | (df['state'] == 'successful') 

df_small = df[(df['state'] == 'successful') | (df['state'] == 'failed')]

cols_to_drop = ['ID', 'name', 'main_category', 'currency', 'deadline', 
                'launched', 'pledged', 'country', 'usd pledged', 
                'usd_pledged_real', 'usd_goal_real']

df_small.drop(columns=cols_to_drop, inplace=True)

target = 'state'
y = df_small[target]
X = df_small.drop(columns=target)

lr_model = make_pipeline(OrdinalEncoder(),
                         StandardScaler(),
                         LogisticRegression(max_iter=100, n_jobs=-1))

lr_model.fit(X, y)


print(df_small['state'].value_counts(normalize=True))
print(lr_model.score(X, y))
print(lr_model.predict([X.iloc[4]])[0])
print(lr_model.predict_proba([X.iloc[4]])[0][1])

dump(lr_model, "lr_model.joblib", compress=True)