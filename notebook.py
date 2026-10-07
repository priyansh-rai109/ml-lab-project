import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data load
df = pd.read_csv("Crop_recommendation.csv")
print("Shape:", df.shape)
print(df.head())

# 2. Data samjho
print(df.info())
print("Missing values:\n", df.isnull().sum())
print(df.describe())

# 3. Crop distribution
df["label"].value_counts().plot(kind="bar", figsize=(10, 4))
plt.title("Crop distribution")
plt.show()
# 4. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop("label", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Feature correlation")
plt.show()

# 5. Boxplot: crop ke hisaab se rainfall
plt.figure(figsize=(12, 5))
sns.boxplot(x="label", y="rainfall", data=df)
plt.xticks(rotation=90)
plt.title("Rainfall by crop")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 6. Features (X) aur target (y) alag karo
X = df.drop("label", axis=1)
y = df["label"]

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# 8. Pehla model: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 10. Teen models ek dictionary mein
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# 11. Sab ko train karo aur accuracy nikalo
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, pred)

# 12. Comparison table
comparison = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
print(comparison.sort_values("Accuracy", ascending=False))
from sklearn.metrics import confusion_matrix, classification_report

# 13. Random Forest ki predictions
y_pred = rf.predict(X_test)

# 14. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
plt.figure(figsize=(12, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

# 15. Classification report
print(classification_report(y_test, y_pred))

# 16. Feature importance
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance)
importance.plot(kind="bar", figsize=(8, 4))
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

import joblib
joblib.dump(rf, "crop_model.pkl")
print("Model saved")