import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#pip install pandas scikit-learn matplotlib
# 1. Load dataset
df = pd.read_csv("sample_-_superstore.csv")

# 2. Select features
X = df[['Sales']]
y = df['Profit']

# 3. Handle missing values (if any)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# 8. Show some predictions
result = pd.DataFrame({
    'Sales': X_test['Sales'],
    'Actual Profit': y_test,
    'Predicted Profit': y_pred
})
print("\nSample Predictions:\n", result.head())

# 9. Visualization
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Sales vs Profit (Linear Regression)")
plt.show()
