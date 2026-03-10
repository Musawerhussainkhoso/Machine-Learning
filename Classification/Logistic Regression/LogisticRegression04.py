from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
X = [
    [20000,1],
    [25000,1],
    [30000,0],
    [35000,1],
    [40000,1],
    [45000,0],
    [50000,1],
    [55000,1]
]

y = [0,0,0,1,1,1,1,1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# User input
income = float(input("Enter monthly income: "))
credit_history = int(input("Enter credit history (1=Good, 0=Bad): "))

# Prediction
prediction = model.predict([[income, credit_history]])[0]

# Result
if prediction == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")