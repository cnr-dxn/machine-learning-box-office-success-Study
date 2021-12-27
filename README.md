# Machine Learning Oriented Box Office Success Study

A study on the major determining factors for box office performance in the film industry. This project analyzed and normalized past box office data utilizing The Movie Database (TMDB API), evaluated trends and patterns within said data, used said data to create models to predict the performance of upcoming films (utilizing machine learning libr, and ultimately determined what attributes dictate success for an upcoming film. 

**Data Collection**: To gather data, a TMDB API was used to collect a set of movies spanning over the following time range 1918-2020. This data was initially collected & stored in a Linode-hosted cloud SQL database, however later converted to a locally stored .csv format to allow for the easiest access from libraries and functions such as pandas, numpy, and sklearn.

**Data Analysis & Preparation**: To prepare the data for model processing and usage, One-Hot Encoding and Bagging of variables was utilized.

**Machine Learning Model Utilization and Evaluation**: To achieve the highest possible prediction rate, an ensemble learning method was used, using both bagging (with a Random Forest Regressor from sklearn) and boosting (with an Extreme Gradient Boosted Regressor from XGBoost), both utilizing a 90/10 train-test split. To tune the model, methods of general normalization (with a log transformation), hyperparameter tuning (with a GridSearchCV model), and feature selection (with an RFECV model). The best model was determined to be the XGBoost model, with an R^2 score of 0.69.
