from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------
# Actual values (Ground Truth)
# ------------------------------
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# ------------------------------
# Predicted values by ML model
# ------------------------------
y_pred = [1, 0, 1, 0, 0, 1, 1, 1, 0, 1]

# ------------------------------
# Model Evaluation Metrics
# ------------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Model Performance Metrics")
print("--------------------------")
print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)

# ------------------------------
# Confusion Matrix
# ------------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ------------------------------
# Detailed Classification Report
# ------------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred))