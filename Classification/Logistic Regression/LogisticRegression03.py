from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
X = [
    [1,50],
    [2,55],
    [3,60],
    [4,65],
    [5,70],
    [6,75],
    [7,80],
    [8,85]
]

y = [0,0,0,0,1,1,1,1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# User input
study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))

# Prediction
prediction = model.predict([[study_hours, attendance]])[0]

# Result
if prediction == 1:
    print("Prediction: Student will PASS")
else:
    print("Prediction: Student will FAIL")