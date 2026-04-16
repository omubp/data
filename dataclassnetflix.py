import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("netflix_titless.csv", encoding='latin1')

# 2. Select important columns
df = df[['type', 'release_year', 'duration_minutes']]

# 3. Clean duration (extract number)
df['duration_minutes'] = df['duration_minutes'].str.extract('(\d+)')
df['duration_minutes'] = pd.to_numeric(df['duration_minutes'], errors='coerce')

# 4. Handle missing values
df = df.dropna()

# 5. Encode target (Movie/TV Show → numbers)
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# 6. Define X and y
X = df[['release_year', 'duration_minutes']]
y = df['type']

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model = GaussianNB()
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)

# 10. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 11. Classification report
print("\nReport:\n", classification_report(y_test, y_pred))

# 12. Show sample predictions
result = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nSample Output:\n", result.head())

# 13. Convert back to labels (important for viva)
print("\nReadable Output:")
for i in range(5):
    actual = le.inverse_transform([y_test.iloc[i]])[0]
    pred = le.inverse_transform([y_pred[i]])[0]
    print(f"Actual: {actual} | Predicted: {pred}")
