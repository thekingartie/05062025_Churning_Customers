# -*- coding: utf-8 -*-
"""05062025_Churning_Customers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KmC_gMKBVCOp5mGZ3BAXJXkB59aI22VW

## ***CHURNING CUSTOMERS IN A TELECOMS COMPANY***

**IMPORTATION OF NECCESSARY LIBRARIES AND MOUNTING OF** **DRIVE**
"""

# Installation of required packages
!pip install tensorflow scikeras scikit-learn
!pip install keras-tuner

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning libraries
import sklearn as sk
from sklearn import metrics
from sklearn.model_selection import (KFold, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, accuracy_score, make_scorer)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Keras
import keras
import keras_tuner
from kerastuner import Objective
from kerastuner.tuners import RandomSearch
from keras.models import Model
from scikeras.wrappers import KerasClassifier
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Additional Libraries
from google.colab import drive

# Mounts the Google Drive to the specified directory '/content/drive'
# This allows access to files and data stored in your Google Drive within the Colab environment
drive.mount('/content/drive')

"""***DATA COLLECTION***"""

# Reasa CSV file into a Pandas DataFrame
# The file 'player_21.csv' is located in the specified Google Drive directory and is loaded into the variabe 'first_dataset' for further data analysis
dataset = pd.read_csv('/content/drive/MyDrive/CustomerChurn_dataset.csv')

dataset

dataset.info()

"""*DEFINING AND DROPPING FEATURES WITH MORE THAN **30%** OF THE DATA MISSING*"""

# Define the percentage threshold
percentage = 30

# Calculating the percentage of missing values in each of the columns
missing_percentage_per_each_column = (dataset.isnull().sum() / len(dataset)) * 100

# Drop columns exceeding the threshold
drop_columns = missing_percentage_per_each_column[missing_percentage_per_each_column > percentage].index
dataset.drop(columns = drop_columns, inplace = True)

dataset

# Analyzing the new DataFrame
dataset.info(verbose = True)

"""***FEATURE ENGINEERING & FEATURE IMPORTANCE***"""

# Non-numeric Splitting (Imputing for Categorical Values)
categorical_columns = dataset.select_dtypes(exclude = ['number'])
categorical_columns.info(verbose = True)

# Numeric Splitting (Imputing for Numerical Values)
numeric_columns = dataset.select_dtypes(include = ['number'])
numeric_columns.info(verbose = True)

"""***IMPUTATION AND ENCODING***"""

# Encoding the Object Values to numeric Data Types
# Using Label Encoder
label_encoder = LabelEncoder()

# Encoding the Categorical Values
for column in categorical_columns.columns:
  categorical_columns[column] = label_encoder.fit_transform(categorical_columns[column])
categorical_columns.head()

# Combining the two DataFrame
combined_dataset = pd.concat([categorical_columns, numeric_columns], axis = 1)
combined_dataset.info()

combined_dataset.head()

# Split target and feature variables
X = combined_dataset.drop("Churn", axis = 1)
y = combined_dataset["Churn"]

# Scaling the data values (to train)
scaling = StandardScaler()
data_scaled = scaling.fit_transform(X)

# Transorm it into a DataFrame
X = pd.DataFrame(data_scaled, columns = X.columns)
X.head()

X.info()

# Splitting the dataset for training and testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Train the model
rfc_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
rfc_model.fit(X_train, y_train)

"""***TRAINING THE MODEL USING FEATURE IMPORTANCE***"""

# Obtaining important features
labeled_feature = X.columns
feature_importance = rfc_model.feature_importances_

# Feature Importance Sorting
feature_importance_dataset = pd.DataFrame({"Feature Displayed": labeled_feature, "Importance": feature_importance})

# Sorted Features (in Descending Order)
feature_importance_dataset = feature_importance_dataset.sort_values(by="Importance", ascending=False)
feature_importance_dataset

# Check the column names in feature_importance_dataset
print(feature_importance_dataset.columns)

# Update the column names in the plot accordingly
plt.bar(feature_importance_dataset['Feature Displayed'], feature_importance_dataset['Importance'])
plt.xlabel("Feature Displayed")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation="vertical")
plt.show()

# Set a threshold for feature importance
threshold = 0.02

# Select features with importance greater than the threshold
important_features = X.columns[feature_importance > threshold]

# Create a new DataFrame with only the selected important features
X_selected = X[important_features]

# Display the selected features
X_selected

X = X_selected

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

"""# **EXPLORATORY DATA ANALYSIS AND DATA PREPROCESSING**"""

# Printing the shape of the dataset
print ("The shape of the dataset : ", X_selected.shape)

# Display descriptive statistics for the DataFrame
print(X_selected.describe())

# Pie Chart for a single categorical variable
plt.figure(figsize = (6, 6))
dataset["Churn"].value_counts().plot(kind = "pie", autopct = "%1.1f%%", startangle = 90, colors = ["blue", "red"])
plt.title("Pie Chart for Churn")
plt.show()

"""*The pie chart above illustrates that less customers who have stayed longer with the company are less likely to move to another company (Churn)*"""

# Assuming "dataset" is the DataFrame
sns.boxplot(x = "Churn", y = "tenure", data = dataset)
plt.title("Relationship between Churn and Tenure")
plt.show()

"""*The graph above illustrates the customers who have stayed longer with the company are less likely to move to another Company (Churn)*"""

# Assuming "Dataset" is your DataFrame
sns.boxplot(x = "Churn", y = dataset["MonthlyCharges"], data = dataset)
plt.title("Relationship between Churn and Monthly Charges")
plt.show()

"""*The graph above suuggest that customers with Higher Monthly Charges are more liekly to move to another Company (Churn)*"""

# List of categorical values for the 'Contract' variable
Categorical = ['One year', 'Two years', 'Month-to-month']

# Create a countplot to visualize the distribution of 'Churn' within each category of 'Contract'
sns.countplot(x=dataset["Contract"], hue=dataset["Churn"], palette="Set2")

# Add title and axis labels to the plot
plt.title("Contract")
plt.xlabel("Contract")
plt.ylabel("Count")

# Add a legend to show the relationship between colors and the 'Churn' variable
plt.legend(title="Churn", loc="upper right", bbox_to_anchor=(1.2, 1))

# Display the plot
plt.show()

"""*The graph above portrays customers that pay through **electric checks** are more likey to move to another company (churn) than those who pay via other methods. Thus, the customers who pay **month-to-month** will church quite a lot*

***MULTI-LAYER PERCEPTION MODEL USING FUNCTIONAL API***
"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

# Function to create a Keras model using the functional API
def create_model(optimizer='adam', hidden_layer1_units=50, hidden_layer2_units=20):
    input_layer = Input(shape=(X_train.shape[1],))
    hidden_layer_1 = Dense(hidden_layer1_units, activation ='relu')(input_layer)
    hidden_layer_2 = Dense(hidden_layer2_units, activation ='relu')(hidden_layer_1)
    output_layer = Dense(1, activation = 'sigmoid')(hidden_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss ='binary_crossentropy', metrics = ['accuracy'])
    return model

# Wrap the Keras model using KerasClassifier
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 32, verbose = 0)

# Keras Functional API model
def create_model(optimizer = 'Adam', hidden_layer1_units = 64, hidden_layer2_units = 32):
  input_layer = Input(shape = (X_train.shape[1],))
  hidden_layer_1 = Dense(hidden_layer1_units, activation = 'relu')(input_layer)
  hidden_layer_2 = Dense(hidden_layer2_units, activation = 'relu')(hidden_layer_1)
  output_layer = Dense(1, activation = 'sigmoid')(hidden_layer_2)

  model = Model(inputs = input_layer, outputs = output_layer)
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
  return model

  # Wrap the Keras model using KerasClassifier
model = KerasClassifier(model = create_model, epochs = 10, batch_size = 32, verbose = 0, hidden_layer1_units = 32, hidden_layer2_units = 16)

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor = "val_loss", restore_best_weights = True )

"""***Hyper paramter tunning using the Keras Tuner***"""

# Define a parameter grid for grid search
param_grid = {
    'optimizer':['adam','sgd','rmsprop'],
    'hidden_layer1_units':[50,100,150],
    'hidden_layer2_units':[20,40,60]
}

auc_scorer = make_scorer(roc_auc_score, greater_is_better = True)
# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = auc_scorer, cv = StratifiedKFold(n_splits = 5), verbose = 1, error_score = 'raise')

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "tensorflow")

grid_result = grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_result.best_params_}')
print(f'Best AUC Score: {grid_result.best_score_}')

#Evaluate the best model on the test set
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_binary = (y_pred>0.5).astype(int)
accuracy_best = accuracy_score(y_test,y_pred_binary)
auc_score_best = roc_auc_score(y_test,y_pred)
print(f'Test Accuracy: {accuracy_best}')
print(f'AUC Score: {auc_score_best}')

"""# ***MODEL OPTIMIZATION***"""

# Train and test the optimized model
optimized_model = grid_search.best_estimator_
optimized_model.fit(X_train, y_train)
optimized_accuracy = optimized_model.score(X_test, y_test)
optimized_auc_score = roc_auc_score(y_test, optimized_model.predict_proba(X_test)[:, 1])

# Print optimized results
print(f'Optimized Model Accuracy: {optimized_accuracy}')
print(f'Optimized AUC Score: {optimized_auc_score}')

"""***SAVING THE MODEL***"""

# Define the model using optimized hyperparameters

input_layer = Input(shape = (X_train.shape[1],))
hidden_layer_1 = Dense(128, activation = 'relu')(input_layer)
hidden_layer_2 = Dense(32, activation = 'relu')(hidden_layer_1)
output_layer = Dense(1, activation = 'sigmoid')(hidden_layer_2)

optimized_model = Model(inputs = input_layer, outputs = output_layer)
optimized_model.compile(optimizer = RMSprop(learning_rate = 0.001, rho = 0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])


optimized_model.fit(X_train, y_train, epochs = 10, batch_size = 32, verbose = 1)

y_optimized_pred = best_model.predict(X_test)
y_pred_optimized_binary = (y_pred > 0.5).astype(int)

optimized_accuracy = accuracy_score(y_test,y_pred_optimized_binary)
optimized_auc_score = roc_auc_score(y_test,y_optimized_pred)

import pickle

# Define the file path where you want to save the model
model = optimized_model

model.save('model.h5')

with open('scaler.pkl', 'wb') as scaler_file:
  pickle.dump(scaler, scaler_file)