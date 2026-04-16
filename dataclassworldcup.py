import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

#pip install pandas scikit-learn matplotlib
# Load only Sheet3
df = pd.read_excel("world_cup_results.xlsx", sheet_name="WorldCups")

# 2. Clean Attendance column
df['Attendance'] = df['Attendance'].astype(str)
df['Attendance'] = df['Attendance'].str.replace('.', '', regex=False)
df['Attendance'] = pd.to_numeric(df['Attendance'], errors='coerce')

# 3. Select features and target
X = df[['GoalsScored', 'QualifiedTeams', 'MatchesPlayed', 'Attendance']]
y = df['Winner']

# 4. Handle missing values
X = X.fillna(X.mean())

# 5. Encode target (Winner → numbers)
le = LabelEncoder()
y = le.fit_transform(y)

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
model = GaussianNB()
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Actual vs Predicted (numeric)
result = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nActual vs Predicted:\n", result)

# 11. Show predictions with team names (important)
print("\nSample Predictions (with team names):")
for i in range(min(5, len(y_pred))):
    actual_team = le.inverse_transform([y_test[i]])[0]
    predicted_team = le.inverse_transform([y_pred[i]])[0]
    print(f"Actual: {actual_team}  |  Predicted: {predicted_team}")

# 12. Classification report (extra for viva)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

