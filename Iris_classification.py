import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Load Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add Species Column
df['species'] = iris.target

# Convert numbers to names
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# Show first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Class Distribution
print("\nClass Distribution:")
print(df['species'].value_counts())

# Histogram
sns.histplot(df['sepal length (cm)'], kde=True)
plt.title("Sepal Length Distribution")
plt.show()

# Pairplot
sns.pairplot(df, hue='species')
plt.savefig("pairplot.png")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Split data
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy =", accuracy)

# Save accuracy
with open("accuracy.txt", "w") as file:
    file.write(f"Accuracy = {accuracy}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Features and Target
X = df.drop('species', axis=1)
y = df['species']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy =", accuracy)

# Save Accuracy
with open("accuracy.txt", "w") as file:
    file.write(f"Accuracy = {accuracy}")

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            cmap='Blues',
            fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()