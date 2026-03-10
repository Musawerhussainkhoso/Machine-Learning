import pandas as pn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Create dataset using dictionary
data = {
    "StudyHours": [1,2,3,4,5,6,7,2,3,4,5,6,7,3,2,5,6,4,3,7],
    "Attendance": [50,55,60,65,70,75,80,58,62,68,72,78,85,64,57,74,79,66,61,88],
    "PreviousMarks": [40,45,50,55,60,65,70,48,52,58,63,67,72,54,46,66,69,57,51,75],
    "Pass": [0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,1]
}

# Convert to DataFrame
df = pn.DataFrame(data)

# Show first rows
print(df.head())

# Check null values
print(df.isnull().sum())

# Features and target
X = df.drop("Pass", axis=1)
y = df["Pass"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))