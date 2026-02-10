import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load the dataset
df = pd.read_csv('suv_data.csv')

# 2. Select features and target
# Based on original code: X = suv_car_df.iloc[:,[2,3]] -> Age, EstimatedSalary
# y = suv_car_df.iloc[:,4] -> Purchased
X = df.iloc[:, [2, 3]]
y = df.iloc[:, 4]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# 4. Feature Scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# 5. Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Save the model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(sc, 'scaler.pkl')

print("Model and Scaler have been saved successfully!")
print(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
print(f"Testing Accuracy: {model.score(X_test_scaled, y_test):.2f}")
