import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load dataset
data = pd.read_csv(r"C:\Users\SAYALI GADGE\Downloads\archive\diabetes_prediction_dataset.csv")
print(data.head())

# Convert categorical data into numbers
data = pd.get_dummies(data, drop_first=True)

# Split input and output
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", acc)
print("Error Rate:", 1-acc)
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))