import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the actual data from the Excel file into a DataFrame
# Replace 'path_to_excel_file.xlsx' with the path to your actual Excel file
df = pd.read_excel('data.xlsx')

# Preprocessing
# Handle missing values represented as -999
df.replace(-999, np.nan, inplace=True)

# Encoding categorical variables (binary)
binary_columns = [
    'sex', 'major_comorbidity', 'chemo_before_liver_resection', 'clinrisk_stratified',
    'extrahep_disease', 'steatosis_yesno', 'presence_sinusoidal_dilata', 'NASH_yesno',
    'NASH_greater_4', 'fibrosis_greater_40_percent', 'vital_status', 'progression_or_recurrence',
    'vital_status_DFS', 'progression_or_recurrence_liveronly', 'vital_status_liver_DFS'
]
df[binary_columns] = df[binary_columns].astype(float)

# Select features and target variable
# Excluding 'id' (Patient-ID) and 'De-identify Scout Name' (De-identifed scout name) from the features
X = df.drop(columns=['Patient-ID', 'De-identify Scout Name', 'progression_or_recurrence'])
y = df['progression_or_recurrence']
# Impute missing values with the median of each column
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Feature selection (select top 5 features based on mutual information)
selector = SelectKBest(mutual_info_classif, k=5)
X_selected = selector.fit_transform(X_imputed, y)
# Get the boolean mask for selected features
selected_features_mask = selector.get_support()

# Get the names of the selected features
selected_feature_names = X.columns[selected_features_mask]
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Train the SVM model
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
report = classification_report(y_test, y_pred)

print(report)
print(selected_feature_names)
