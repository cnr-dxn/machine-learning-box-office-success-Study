import pandas as pd
from sklearn.model_selection import train_test_split
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
counter = 100;
while counter < 100:
    model = RandomForestClassifier(n_estimators=40, max_depth=counter)
    print("- split it fully...");
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['revenue'], test_size=0.25, random_state=42)

    print("- fitting...");
    model.fit(X_train, y_train);
    print("- predicting...");
    y_pred = model.predict(X_test)

    score = r2_score(y_test, y_pred)

    print("  ", counter, ": ", score, sep='')
    counter = counter + 10;

model = RandomForestClassifier(n_estimators=40, max_depth=None)
print("- split it fully...");
X_train, X_test, y_train, y_test = train_test_split(df[features], df['revenue'], test_size=0.25, random_state=42)

print("- fitting...");
model.fit(X_train, y_train);
print("- predicting...");
y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

print("  ", counter, ": ", score, sep='')
counter = counter + 10;
#model = RandomForestRegressor(n_estimators= 76)

#print("split it fully...");
#X_train, X_test, y_train, y_test = train_test_split(df[features], df['revenue'], test_size=0.25, random_state=42)

#print("fitting...");
#model.fit(X_train, y_train);
#
#print("predicting...");
#y_pred = model.predict(X_test)

#score = r2_score(y_test, y_pred)
#print("Acuuracy:", score)
