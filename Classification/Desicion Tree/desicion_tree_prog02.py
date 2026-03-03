# Roll No: 115

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Create dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8],
    "SleepHours": [8, 7, 6, 6, 5, 5, 4, 3],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Step 3: Separate features (X) and target (y)
X = df[["StudyHours", "SleepHours"]]
y = df["Pass"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Create Decision Tree model
model = DecisionTreeClassifier()

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Predict new data
new_data = [[5, 6]]  # StudyHours=5, SleepHours=6
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")