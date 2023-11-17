# 05062025_Churning_Customers

This project focuses on developing a deep learning application to predict Customer Churn in a Telecom Company. 

## Table of Contents

### Introduction
- Data Collection
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Selection and Engineering
- Data Scaling and Splitting
- Multi-Layer Perceptron (MLP) Model using Functional API
- Model Evaluation
- Model Optimization
- Saving the Final Model

  

## Introduction

Customer churn is a critical concern for large companies, particularly in the telecom industry, as it directly impacts revenue. The goal is to identify factors that contribute to customer churn and build a predictive model to assist telecom operators in anticipating potential churn.

## Data Collection

The dataset is loaded from a specified Google Drive directory, containing information about customer attributes and churn status.

`dataset_link = "/content/drive/MyDrive/CustomerChurn_dataset.csv"`

`dataset = pd.read_csv(dataset_link)`


## Data Preprocessing
Columns with more than 30% missing data are dropped, and the dataset is split into numeric and categorical features for further analysis.

`threshold = 30`

`dataset.drop(columns = missing_percentage_per_each_column[missing_percentage_per_each_column > threshold].index, inplace = True)`


## Exploratory Data Analysis (EDA)
EDA involves visualizing relationships between features and churn, such as tenure, monthly charges, and contract types, to understand patterns.

# Sample Graph Plots
`sns.boxplot(x = "Churn", y = "tenure", data = dataset)`

`sns.boxplot(x = "Churn", y = "MonthlyCharges", data = dataset)`

`sns.countplot(x = dataset["Contract"], hue = dataset["Churn"], palette = "Set2")`


# Feature Selection and Engineering
Feature selection is performed based on feature importance obtained from a Random Forest Classifier.

`important_features = X.columns[feature_importance > threshold]`

`X_selected = X[important_features]`


# Data Scaling and Splitting
The dataset is scaled, and the data is split into training and testing sets.

`scaler = StandardScaler()`

`X_scaled = scaler.fit_transform(X)`

`X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)`


## Multi-Layer Perceptron (MLP) Model using Functional API
A Keras MLP model is created using the Functional API, with hyperparameter tuning..

`def create_model(optimizer = 'adam', hidden_layer1_units = 50, hidden_layer2_units = 20):`

   `# Model architecture definition`
   
   `input_layer = Input(shape = (X_train.shape[1],))`
   
   `hidden_layer_1 = Dense(hidden_layer1_units, activation = 'relu')(input_layer)`
   
   `hidden_layer_2 = Dense(hidden_layer2_units, activation = 'relu')(hidden_layer_1)`
   
   `output_layer = Dense(1, activation = 'sigmoid')(hidden_layer_2)`

   `model = Model(inputs = input_layer, outputs = output_layer)`
   
   `model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])`
   
   `return model`


`model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 32, verbose = 0)`


## Model Evaluation
# AUC Score and Accuracy
The model is evaluated using AUC score and accuracy.

`y_pred = best_model.predict(X_test)`

`y_pred_binary = (y_pred > 0.5).astype(int)`

`accuracy_best = accuracy_score(y_test, y_pred_binary)`

`auc_score_best = roc_auc_score(y_test, y_pred)`


## Model Optimization
Hyperparameter tuning is performed using GridSearchCV for optimizing the model's performance..

`grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = auc_scorer, cv = StratifiedKFold(n_splits = 5), verbose = 1, error_score = 'raise')`

`grid_result = grid_search.fit(X_train, y_train)`

`optimized_model = grid_search.best_estimator_`


# Saving the Final Model
The optimized model is saved using pickle for future predictions.

`with open('model.pkl', 'wb') as file:`

    `pickle.dump(optimized_model, file)`


`with open('scaler.pkl', 'wb') as scaler_file:`


    `pickle.dump(scaler, scaler_file)`
