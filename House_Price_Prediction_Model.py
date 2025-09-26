import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns


# Using Pandas to Clean Data

# the house price dataset
house_price_dataset = pd.read_csv('Housing.csv')
house_price_dataframe = pd.DataFrame(house_price_dataset)

# check for nan values
house_price_dataframe.isnull().sum()

label_encoder = LabelEncoder()

# Apply Label Encoding to binary categorical columns
for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
    house_price_dataframe[col] = label_encoder.fit_transform(house_price_dataframe[col])

# Removing furnishstatus
house_price_dataframe.drop(['furnishingstatus'], axis = 'columns', inplace = True)

# Create new features
house_price_dataframe['price_per_sqft'] = house_price_dataframe['price'] / house_price_dataframe['area']
house_price_dataframe['rooms_per_bathroom'] = house_price_dataframe['bedrooms'] / house_price_dataframe['bathrooms']



# Constructing Heatmap
'''
correlation = house_price_dataframe.corr()

plt.figure(figsize = (10, 10))
sns.heatmap(correlation, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws = {'size': 8}, cmap = 'Blues')
plt.show()
'''



# Utilizing ML

# Splitting the data into target and training data
X = house_price_dataframe.drop(['price'], axis = 1)
Y = house_price_dataframe['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 2)

# Hyperparameter using GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}
     
# Fit the best model 
random_search = RandomizedSearchCV(estimator=XGBRegressor(objective='reg:squarederror'), param_distributions=param_grid, n_iter= 10, scoring='neg_mean_absolute_error', cv=5)
random_search.fit(X_train, Y_train)

# Get the best model 
best_model = random_search.best_estimator_



# Model Training For Train Data

# Accuracy for prediction on training data
training_data_prediction = best_model.predict(X_train)

# R squared error
train_r_square = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
train_mae = metrics.mean_absolute_error(Y_train, training_data_prediction)

# Mean Absolute Percentage Error
train_mpe = metrics.mean_absolute_percentage_error(Y_train, training_data_prediction)


# Model Training For Testing Data

# Accuracy for prediction on training data
testing_data_prediction = best_model.predict(X_test)

# R squared error
test_r_square = metrics.r2_score(Y_test, testing_data_prediction)

# Mean Absolute Error
test_mae = metrics.mean_absolute_error(Y_test, testing_data_prediction)

# Mean Absolute Percentage Error
test_mpe = metrics.mean_absolute_percentage_error(Y_test, testing_data_prediction)


'''
# Constructing a scatter plot (helps show relationship)
plt.scatter(Y_train, training_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.show()
'''