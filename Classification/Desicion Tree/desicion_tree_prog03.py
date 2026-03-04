import pandas as pn 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

read_file = pn.read_excel("Decision_Tree_Practice_Dataset.xlsx")
copy_file = read_file.copy()

print(copy_file.head())
print(copy_file.info())
print(copy_file.isnull().sum())
copy_file.fillna(copy_file.mean(), inplace=True)
print(copy_file.isnull().sum())

X = copy_file.drop("Pass",axis = 1)
y = copy_file["Pass"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

inp_01 = float(input("Enter Study Hours: "))
inp_02 = float(input("Enter Sleep Hours: "))
inp_03 = float(input("Enter Attendance Percentage: "))
inp_04 = float(input("Enter Previous Marks: "))
new_data = [[inp_01, inp_02, inp_03, inp_04]]
prediction = model.predict(new_data)[0]
if prediction == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")  

