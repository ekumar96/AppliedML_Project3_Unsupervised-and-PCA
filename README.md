# Project 2 - Spring 2022

Applied ML Spring 2022 Project 2: Trees and Calibration

Eshan Kumar, ek3227

This project consists of Four parts. In all parts of the project, I work on a dataset of the performance of multiple players in a video game, CS:GO. Using this data, I try to predict whether a given player will win or not. The data can be found here: https://www.kaggle.com/gamersclub/brazilian-csgo-plataform-dataset-by-gamers-club?select=tb_lobby_stats_player.csv

# Part 1 - Decision Tree
In the first part of the project, I pre-process the data and implement a Decision Tree Classifier. 

### Data Cleaning
First, I calculate which columns contain missing data, and determine if any of them have so much missing data that they should be removed. Seeing that no columns have a large amount of missing data, I instead impute the missing values by replacing missing values with the mean of a continuous feature and the most common category for a categorical feature. 

### Data Visualization
I then plot the relationships between the label (Winning) and the continuous variables on boxplots, and the relationships between the categorical feature and the label in a bar plot. This way I can note any positive correlations. 

### Data Preprocessing
Next, I split the data into 80% development data and 20% test data. I then one-hot encode categorical data for both the development and test set. I make sure to fit the OneHotEncoder on the development set, in case the test set's categorical feature is missing a category. I don't scale the data because this is not needed for Decision Trees.

### Decision Tree Implementation
I fit and visualize a default decision tree classifier, which automatically trains until all leaves are pure. This means that the tree is overfit on the training set, and has a perfect score on the training set with a lower score on the test set. 

### Decision Tree Pruning (ccp_alpha)
I then prune the decision tree using the ccp_alpha parameter, doing a hyperparameter search on this value in order to create a model that performs well on the test set. 

### Feature Evaluation
I use the weights to evaluate which features seem to be the biggest factors in determining whether a player wins CS:GO. 

# Part 2 - Random Forest
### Random Forest Implementation
I fit a random forest, noting it's higher score because it is an ensemble method. I then verify that every tree in the forest has completely pure leaves. 

### Hyperparameter Tuning (estimators, ccp_alpha)
I create a search space across estimators (number of trees in the forest) and ccp_alphas (controls pruning/leaf purity), and do a grid search with a 5-fold cross validation in this space in order to determine an optimal combination of hyperparameters. I then evaluate the accuracy of some models with different hyperparameters on the Test set, and find that it is more accurate. 

### Feature Evaluation
I use the weights to evaluate which features seem to be the biggest factors in determining whether a player wins CS:GO. 

# Part 3 - Gradient Boosted Trees
### Gradient Boosting Classifier, HistGradientBoostingClassifier, and XGBoost Classifier Implementation and Hyperparameter Tuning (learning_rate, estimators, lambda, l2_regularization, min_impurity_decrease)
I fit these models, tune their hyperparameters score them on the test set. I then compare all of these models to each other and to the Random Forest and Decision Tree in terms of performance on the test set. I find that the HistGradientBoostingClassifier performed the best, with an accuracy of 0.8.

### Hyperparameter Tuning (estimators, ccp_alpha)
I create a search space across estimators (number of trees in the forest) and ccp_alphas (controls pruning/leaf purity), and do a grid search with a 5-fold cross validation in this space in order to determine an optimal combination of hyperparameters. I then evaluate the accuracy of some models with different hyperparameters on the Test set, and find that it is more accurate. 

### Feature Evaluation and retraining
I use the weights from the XGBoost Classifier to evaluate which features seem to be the biggest factors in determining whether a player wins CS:GO. Using the top 7 important features, I retrain the model with the XGBoost model and find that the accuracy has remained very similar and the model trained much faster even though the data size decreased significantly. 

# Part 4 - Model Calibration
### Calibration Diagnostics
I estimated the XGBoost model Brier Score and plotted its Calibration curve. 

### Calibration Methods (Platt scaling, Isotonic Regression)
I used Platt scaling and Isotonic Regression to calibrate the models, and determined the brier scores and evaluated the calibration curves after calibration.
