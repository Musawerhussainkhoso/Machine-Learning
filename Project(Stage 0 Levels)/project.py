import pandas as pn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
read_file = pn.read_excel("Diabetes_Prediction_Dataset.xlsx")
copy_file = read_file.copy()
print(copy_file.head())
print(copy_file.info())
print(copy_file.isnull().sum())
X = copy_file.drop("Outcome", axis=1)  # drop target column → features
y = copy_file["Outcome"]                # target column only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
Ss = StandardScaler()
X_train = Ss.fit_transform(X_train)
X_test = Ss.transform(X_test)
model = LogisticRegression()
model.fit(X_train , y_train)
inp_01 = int(input("Enter Pregnancies:"))
inp_02 = int(input("Enter Glucose:"))
inp_03 = int(input("EnterBloodPressure:"))
inp_04 = int(input("Enter SkinThickness:"))
inp_05 = int(input("Enter Insulin:"))
inp_06 = float(input("Enter BMI:"))
inp_07 = float(input("Enter DiabetesPedigreeFunction:"))
inp_08 = int(input("Enter Age:"))
userinput = pn.DataFrame(
    [[inp_01, inp_02, inp_03, inp_04, inp_05, inp_06, inp_07, inp_08]],
    columns=X.columns
)
user_input_scaled = Ss.transform(userinput)
prediction = model.predict(user_input_scaled)[0]
if prediction == 0:
    print("The person is not diabetic:")
else:
    print("The person is diabetic:")    
