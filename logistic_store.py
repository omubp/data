import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset

df = pd.read_csv("sample_-_superstores.csv", encoding='latin1')
print(df.head())

# 2. Create target column (Profit → 1, Loss → 0)
df['Profit_Label'] = df['Profit'].apply(lambda x: 1 if x > 0 else 0)

# 3. Select features and target
X = df[['Sales']]
y = df['Profit_Label']

# 4. Handle missing values
X = X.fillna(X.mean())

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 11. Show some predictions
result = pd.DataFrame({
    'Sales': X_test['Sales'],
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nSample Predictions:\n", result.head())
