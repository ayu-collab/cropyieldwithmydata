import pandas as pd

# Load the dataset
data = pd.read_csv('crop_data.csv')

# Print the first few rows
print(data.head())



# 1. Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# 2. Encode categorical data (if needed)
data['soil_type'] = data['soil_type'].astype('category').cat.codes

# 3. Separate features (X) and target (y)
X = data[['rainfall', 'temperature', 'soil_type']]  # Input features
y = data['yield']  # Target variable

print("Features (X):")
print(X.head())

print("Target (y):")
print(y.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Print the coefficients and intercept
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)



from sklearn.metrics import mean_squared_error, r2_score

# 1. Make predictions using the testing set
y_pred = model.predict(X_test)

# 2. Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared Score (R2):", r2)


# Example of new, unseen data
new_data = pd.DataFrame({
    'rainfall': [110, 95],
    'temperature': [22, 30],
    'soil_type': [0, 1]
})

# Predict the yield for the new data
new_predictions = model.predict(new_data)

print("Predicted yields for new data:", new_predictions)



import joblib

# Save the model to a file
joblib.dump(model, 'crop_yield_model.pkl')
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load('crop_yield_model.pkl')

# Use the loaded model to make predictions
predictions = loaded_model.predict(new_data)
print("Predictions from the loaded model:", predictions)




import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot: Rainfall vs Yield
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='rainfall', y='yield')
plt.title('Rainfall vs Yield')
plt.show()

# Scatter plot: Temperature vs Yield
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='temperature', y='yield')
plt.title('Temperature vs Yield')
plt.show()



# Compare predictions with actual yields
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line for perfect prediction
plt.show()

