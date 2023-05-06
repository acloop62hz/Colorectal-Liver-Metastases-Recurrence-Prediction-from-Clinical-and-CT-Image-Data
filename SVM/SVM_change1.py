import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#This function imputes missing values in the data with the median value of each column. 
# It returns the data with imputed values.
#data: A numpy array where each row represents an instance and each column represents a feature. Missing values are represented by NaN.
def simple_imputer(data):
    col_medians = np.nanmedian(data, axis=0)
    data_imputed = np.where(np.isnan(data), col_medians, data)
    return data_imputed
def normalize_data(data):
    # Normalize the data by scaling each feature to the range [0, 1]
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    data_normalized = (data - min_vals) / (max_vals - min_vals)
    return data_normalized

def normalized_feature_selection(data, k):
    # Normalize the data using the custom normalization function
    data_normalized = normalize_data(data)
    
    # Calculate the variance of each feature in the normalized data
    feature_variances = np.var(data_normalized, axis=0)
    
    # Sort the indices of the variances in ascending order
    # and then select the last 'k' indices to get the top 'k' variances
    top_k_indices = np.argsort(feature_variances)[-k:]
    
    # Use the selected indices to extract the top 'k' features from the normalized data
    data_selected = data_normalized[:, top_k_indices]
    
    return data_selected, top_k_indices
#This function selects the top k features based on feature variance.
# It returns the data with only the selected features and the indices of the selected features.

def feature_selection(data, target, k):
    # Calculate the correlation coefficients between each feature and the target variable
    correlations = np.array([np.corrcoef(data[:, i], target)[0, 1] for i in range(data.shape[1])])
    
    # Get the absolute values of the correlation coefficients
    abs_correlations = np.abs(correlations)
    
    # Select the top 'k' features with the highest absolute correlation coefficients
    top_k_indices = np.argsort(abs_correlations)[-k:]
    data_selected = data[:, top_k_indices]
    
    return data_selected, top_k_indices
#This function standardizes the features by subtracting the mean and dividing by the standard deviation of each column. 
# It returns the standardized data.


def plot_feature_importance(data, target):
    # Extract feature names from the columns of the input data
    feature_names = data.columns
    
    # Convert the data DataFrame to a NumPy array
    data_array = data.values
    
    # Calculate the correlation coefficients between each feature and the target variable
    correlations = np.array([np.corrcoef(data_array[:, i], target)[0, 1] for i in range(data_array.shape[1])])
    
    # Get the absolute values of the correlation coefficients
    abs_correlations = np.abs(correlations)
    
    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, abs_correlations)
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation Coefficient')
    plt.title('Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



def standard_scaler(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_scaled = (data - mean) / std
    return data_scaled
# trains a Support Vector Machine (SVM) model on the given data using the specified hyperparameters. 
# It returns the weights and bias of the trained model.

def train_svm(X, y, epochs, learning_rate, C=1, threshold=0.5):
    weights = np.zeros(X.shape[1])
    bias = 0
    
    # Convert target variable y to have labels of -1 and 1
    y = np.where(y == 0, -1, 1)
    
    for epoch in range(epochs):
        linear_output = np.dot(X, weights) + bias
        # Apply threshold to the sigmoid function of the linear output
        predictions = np.where(1 / (1 + np.exp(-linear_output)) >= threshold, 1, -1)
        
        for i in range(X.shape[0]):
            # Update weights and bias only when prediction is incorrect
            if y[i] != predictions[i]:
                weights += learning_rate * y[i] * X[i]
                bias += learning_rate * y[i]
            weights -= learning_rate * C * weights
        
        # Print the values of the weights and bias for monitoring
        
    
    return weights, bias

def predict_svm(X, weights, bias, threshold=0.5):
    # Compute the linear output for each sample
    linear_output = np.dot(X, weights) + bias
    
    # Apply threshold to the sigmoid function of the linear output
    # If the sigmoid(linear_output) is greater than or equal to threshold, predict class 1; otherwise, predict class -1
    predictions = np.where(1 / (1 + np.exp(-linear_output)) >= threshold, 1, 0)
    
    return predictions

def cross_validation(X, y, k, epochs, learning_rate, C=1):
    # Initialize a list to store accuracy scores for each fold
    fold_accuracies = []
    
    # Calculate the number of samples per fold
    samples_per_fold = len(X) // k
    
    # Loop through each fold
    for i in range(k):
        # Determine the indices for the test set
        test_indices = list(range(i * samples_per_fold, (i + 1) * samples_per_fold))
        
        # Determine the indices for the training set
        train_indices = list(set(range(len(X))) - set(test_indices))
        
        # Split the data into training and test sets
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Train the SVM model on the training set
        weights, bias = train_svm(X_train, y_train, epochs, learning_rate, C)
        
        # Make predictions on the test set
        predictions = predict_svm(X_test, weights, bias)
        
        # Calculate the accuracy for this fold
        fold_accuracy = np.mean(predictions == y_test)
        fold_accuracies.append(fold_accuracy)
    
    # Calculate the mean accuracy as the final accuracy
    mean_accuracy = np.mean(fold_accuracies)
    return mean_accuracy

def cross_validation_ConfusionMatrix(X, y, k, epochs, learning_rate, C=1):
    # Initialize a list to store accuracy scores for each fold
    fold_accuracies = []
    # Initialize an array to store confusion matrices for each fold
    fold_cm = np.zeros((2, 2))
    
    # Calculate the number of samples per fold
    samples_per_fold = len(X) // k
    
    # Loop through each fold
    for i in range(k):
        # Determine the indices for the test set
        test_indices = list(range(i * samples_per_fold, (i + 1) * samples_per_fold))
        
        # Determine the indices for the training set
        train_indices = list(set(range(len(X))) - set(test_indices))
        
        # Split the data into training and test sets
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Train the SVM model on the training set
        weights, bias = train_svm(X_train, y_train, epochs, learning_rate, C)
        
        # Make predictions on the test set
        predictions = predict_svm(X_test, weights, bias)
        
        # Calculate the accuracy for this fold
        fold_accuracy = accuracy_score(y_test, predictions)
        fold_accuracies.append(fold_accuracy)
        
        # Compute the confusion matrix for this fold
        for j in range(len(y_test)):
            fold_cm[int(y_test[j]), int(predictions[j])] += 1
    
    # Calculate the mean accuracy as the final accuracy
    mean_accuracy = np.mean(fold_accuracies)
    # Compute the average confusion matrix
    #avg_cm = fold_cm / k
    avg_cm = fold_cm 
    return mean_accuracy, avg_cm
#calculates the accuracy score based on the true labels and predicted labels. 
# It returns the accuracy score as a float.
def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Input arrays must have the same length"
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


# Load data from the Excel file into a DataFrame
data = pd.read_excel('e:/ML/data.xlsx')

data = data.loc[~(data == -999).any(axis=1)]

# Select numerical columns
num_cols = [col for col in data.select_dtypes(include=np.number).columns if data[col].nunique() > 2]

# Standardize numerical columns
for col in num_cols:
    col_mean = data[col].mean()
    col_std = data[col].std()
    data[col] = (data[col] - col_mean) / col_std

# Remove the '%' symbol and convert to float for specific columns if they are of string datatype
for col in ['total_response_percent', 'necrosis_percent', 'fibrosis_percent', 'mucin_percent']:
    if data[col].dtype == 'object':
        data[col] = data[col].str.rstrip('%').astype(float)

# Drop columns if they exist in the DataFrame
columns_to_drop = ['Patient-ID', 'De-identify Scout Name']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

if 'progression_or_recurrence_liveronly' in data.columns:
    X = data.drop('progression_or_recurrence_liveronly', axis=1)
else:
    print("Column 'progression_or_recurrence_liveronly' not found in the DataFrame.")
    # Handle the case when the column is missing, e.g., set X = data or use a different column name
y = data['progression_or_recurrence_liveronly']

# Get the names of the columns for feature selection
feature_names = X.columns

# Impute missing values
X_imputed = simple_imputer(X.values)

######################################################################################
################## task of iterate K features, choose optimal K，line graph###########
######################################################################################

# Initialize lists to store accuracy scores, number of features, and selected features for each iteration
accuracies = []
num_features_list = []
selected_features_list = []
optimal_features = None
max_accuracy = 0
optimal_features = None  # Initialize optimal_features as None
optimal_k = None  # Initialize optimal_k as None

# Iterate over the number of features from 5 to the maximum number of columns
for k in range(5, X.shape[1] + 1):
    # Feature selection (select top k features)
    X_selected, selected_indices = feature_selection(X_imputed, y.values, k)

    # Standardize the features
    X_scaled = standard_scaler(X_selected)

    # Perform cross-validation and get the mean accuracy as the final accuracy
    mean_accuracy = cross_validation(X_scaled, y.values, k=5, epochs=100, learning_rate=0.01)
    print(f"Number of Features: {k}, Mean Accuracy: {mean_accuracy}")
    
    # Append the accuracy, number of features, and selected features to the respective lists
    accuracies.append(mean_accuracy)
    num_features_list.append(k)
    selected_features_list.append(feature_names[selected_indices])
    
    # Store optimal features based on maximum accuracy
    if mean_accuracy > max_accuracy:
        max_accuracy = mean_accuracy
        optimal_features = feature_names[selected_indices]
        optimal_k = k 
# Print the selected features, number of features, and accuracies for each iteration
for i in range(len(num_features_list)):
    print(f"Number of Features: {num_features_list[i]}, Selected Features: {selected_features_list[i]}, Accuracy: {accuracies[i]}")
# Print the number of features (k) and selected features list with the highest accuracy
print(f"Optimal Number of Features (k): {optimal_k}")
print(f"Selected Features with Highest Accuracy: {optimal_features}")
print(f"Highest Accuracy: {max_accuracy}")

# Plot the line graph of accuracy vs. number of features
plt.plot(num_features_list, accuracies)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Features')
plt.grid(True)
plt.show()

######################################################################################
################## task of confusion matrix for optimal feautures selected ###########
######################################################################################


# Select only the columns specified in the selected_features_list
X = data[optimal_features].to_numpy()
y =data['progression_or_recurrence_liveronly'].to_numpy()

# Impute missing values and standardize the features
X_imputed = simple_imputer(X)
X_scaled = standard_scaler(X_imputed)

# Perform cross-validation and get the mean accuracy and average confusion matrix
mean_accuracy, avg_cm = cross_validation_ConfusionMatrix(X_scaled, y, k=5, epochs=100, learning_rate=0.01)

# Display the average confusion matrix
sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Average Confusion Matrix')
plt.show()
# Display the mean accuracy score
print(f"Mean Accuracy Score: {mean_accuracy:.4f}")


######################################################################################
################## task of improtance of every feature ###############################
######################################################################################

X = data.drop('progression_or_recurrence_liveronly', axis=1)
plot_feature_importance(X, y)