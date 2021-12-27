import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import RFE, RFECV
# from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler
print("compiled")

print("reading the file...")
# sc = StandardScaler() # Maybe we want to bin continuos data like budget
df = pd.read_excel('TMDB_processed.xlsx')

print("dropping the title...")
df = df.drop(['title'], axis=1)

print("spliting into features...")
features = df.dtypes[(df.columns != 'revenue')].index # Grab all features except that which we are trying to predict

print("split it fully...");
X_train, X_test, y_train, y_test = train_test_split(df[features], df['revenue'], test_size=0.25, random_state=42)

print("creating the model...");
model = RandomForestRegressor()

## this is where all of the model code will go ya know
print("creating the params...");
params = {
	'bootstrap': [False],
	'max_features': ['sqrt'],
	'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
	'n_estimators': [53],
	'min_samples_leaf': [1, 2, 4]
}

print("creating the grid search...");
clf = GridSearchCV(estimator = model, param_grid = params, scoring = 'r2', verbose = 2);

print("creating the fit...");
clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)

#print("fitting...");
#model.fit(X_train, y_train);
#print("predicting...");
#y_pred = model.predict(X_test)

#score = r2_score(y_test, y_pred)

#print("Acuuracy:", score)
