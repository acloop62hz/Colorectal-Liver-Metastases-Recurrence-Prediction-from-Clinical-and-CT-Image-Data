import numpy as np
import pandas as pd

def simple_imputer(data):
    col_medians = np.nanmedian(data, axis=0)
    data_imputed = np.where(np.isnan(data), col_medians, data)
    return data_imputed

def feature_selection(data, target, k):
    feature_variances = np.var(data, axis=0)
    top_k_indices = np.argsort(feature_variances)[-k:]
    data_selected = data[:, top_k_indices]
    return data_selected, top_k_indices

def standard_scaler(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_scaled = (data - mean) / std
    return data_scaled

def train_svm(X, y, epochs, learning_rate, C=1):
    weights = np.zeros(X.shape[1])
    bias = 0
    for epoch in range(epochs):
        linear_output = np.dot(X, weights) + bias
        predictions = np.where(linear_output >= 0, 1, -1)
        for i in range(X.shape[0]):
            if y[i] * predictions[i] <= 0:
                weights += learning_rate * y[i] * X[i]
                bias += learning_rate * y[i]
            weights -= learning_rate * C * weights
    return weights, bias

def predict_svm(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    predictions = np.where(linear_output >= 0, 1, 0)
    return predictions

# Load data from the Excel file into a DataFrame
df = pd.read_excel('data.xlsx')

# Preprocessing (handle missing values, encode categorical variables, etc.)
df.replace(-999, np.nan, inplace=True)
binary_columns = [
    'sex', 'major_comorbidity', 'chemo_before_liver_resection', 'clinrisk_stratified',
    'extrahep_disease', 'steatosis_yesno', 'presence_sinusoidal_dilata', 'NASH_yesno',
    'NASH_greater_4', 'fibrosis_greater_40_percent', 'vital_status', 'progression_or_recurrence',
    'vital_status_DFS', 'progression_or_recurrence_liveronly', 'vital_status_liver_DFS'
] # Define binary columns as before
df[binary_columns] = df[binary_columns].astype(float)


# Select features and target variable
# Excluding 'Patient-ID' and 'De-identify Scout Name' from the features
X = df.drop(columns=['Patient-ID', 'De-identify Scout Name', 'progression_or_recurrence']).to_numpy()
y = df['progression_or_recurrence'].to_numpy()

# Get the names of the columns for feature selection
feature_names = df.drop(columns=['Patient-ID', 'De-identify Scout Name', 'progression_or_recurrence']).columns

# Impute missing values
X_imputed = simple_imputer(X)

# Feature selection (select top 5 features)
X_selected, selected_indices = feature_selection(X_imputed, y, k=5)

# Standardize the features
X_scaled = standard_scaler(X_selected)

# Train and evaluate the SVM model
epochs = 100
learning_rate = 0.01
weights, bias = train_svm(X_scaled, y, epochs, learning_rate)
predictions = predict_svm(X_scaled, weights, bias)
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)
print("Selected features:", feature_names[selected_indices].values)