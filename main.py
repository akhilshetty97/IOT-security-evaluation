import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Read the csv using pandas
sec_score_df = pd.read_csv("new_data5.csv",index_col=0)

# Checking for presence of null values
sec_score_df.isnull().values.any() #False

# Convert categorical variables to numeric
label_encoders = {}
for column in ['os_var', 'os_version_var', 'secure_boot_var', 'protocol_var', 'ids_present', 'segmentation_present', 'input_data_var', 'interaction_data_var', 'rfid_data_var' ]:
    le = LabelEncoder()
    sec_score_df[column] = le.fit_transform(sec_score_df[column])
    label_encoders[column] = le

#Drop open_port_var and password columns
sec_score_df = sec_score_df.drop(columns=['open_port_var','password_var'])

# Separate features and target variable
X = sec_score_df.drop(['es_score'], axis=1)
y = sec_score_df['es_score']

# Split training and test into 70/30 percent respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

#Used GridSearchCV to find the best hyperparameters

# Set up a more constrained parameter grid for faster hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the model with the best parameters
best_rfr = grid_search.best_estimator_
best_rfr.fit(X_train, y_train)

# Train the model with the best parameters
best_params = {
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 300
}

best_rfr = RandomForestRegressor(
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    n_estimators=best_params['n_estimators'],
    random_state=42
)

# Fit the model on the training data
best_rfr.fit(X_train, y_train)

# Evaluate the model
score = best_rfr.score(X_test, y_test)
print(f"Model R^2 Score: {score:.2f}")

y_pred = best_rfr.predict(X_test)

print(f"Mean absolute error: {mean_absolute_error(y_pred,y_test)}")
print(f"Mean squared error: {mean_squared_error(y_pred, y_test)}")
print(f"R2 score: {r2_score(y_pred, y_test)}")

# # Example of input data
# new_data = pd.DataFrame({
#     'os_var': ['Linux'],
#     'os_version_var': ['22.6'],
#     'secure_boot_var': ['Enabled'],
#     'protocol_var': ['HTTP'],
#     'ids_present': ['Yes],
#     'detection_rate': [100],
#     'false_negative_rate': [0],
#     'response_time': [1],
#     'segmentation_present': ['No],
#     'input_data_var': ['Normal input data'],
#     'interaction_data_var': ['Normal input data'],
#     'rfid_data_var': ['Normal input data']
# })

# User input
os_var = input("Enter OS (e.g., Linux): ")
os_version_var = input("Enter OS version (e.g., 22.6): ")
secure_boot_var = input("Is Secure Boot Enabled? (Yes/No): ")
protocol_var = input("Enter Protocol (e.g., HTTP): ")
ids_present = input("Is IDS Present? (Yes/No): ")
detection_rate = int(input("Enter Detection Rate (0-100): "))
false_negative_rate = int(input("Enter False Negative Rate (0-100): "))
response_time = int(input("Enter Response Time (0-100): "))
segmentation_present = input("Is Segmentation Present? (Yes/No): ")
input_data_var = input("Enter Input Data Type: ")
interaction_data_var = input("Enter Interaction Data Type: ")
rfid_data_var = input("Enter RFID Data Type: ")

# Create a DataFrame for the new input data
new_data = pd.DataFrame({
    'os_var': [os_var],
    'os_version_var': [os_version_var],
    'secure_boot_var': [secure_boot_var],
    'protocol_var': [protocol_var],
    'ids_present': [ids_present],
    'detection_rate': [detection_rate],
    'false_negative_rate': [false_negative_rate],
    'response_time': [response_time],
    'segmentation_present': [segmentation_present],
    'input_data_var': [input_data_var],
    'interaction_data_var': [interaction_data_var],
    'rfid_data_var': [rfid_data_var]
})


# Function to handle unseen labels
def handle_unseen_labels(encoder, value):
    if value not in encoder.classes_:
        # Append the new class to the classes_ attribute
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

# Convert categorical variables to numeric using label encoders
for column in new_data.columns:
    if column in label_encoders:
        le = label_encoders[column]
        new_data[column] = new_data[column].apply(lambda x: handle_unseen_labels(le, x))

# Predict with the trained model
prediction_user_input = best_rfr.predict(new_data)

print(f"The predicted es_score is: {prediction_user_input[0]:.2f}")

# Plot the important features
importances = best_rfr.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()

# Predicted vs actual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.grid(True)
plt.show()