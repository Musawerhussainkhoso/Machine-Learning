from sklearn.neighbors import KNeighborsClassifier
#A small clinic records whether patients have diabetes (1) or not (0) based on age and BMI.
X = [
    [25,22.0],
    [30,28.0],
    [35,30.0],
    [40,26.0],
    [45,32.0],
    [50,35.0]
]
y = [0,0,1,0,1,1] #0 = No Diabetes , 1 = Diabetes
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X,y)
age = int(input("Enter the age of the patient: "))
BMI = float(input("Enter the BMI of the patient:"))
predicted_result = model.predict([[age,BMI]])[0]
if predicted_result == 1:
    print("The patient is likely to have diabetes.")
else:
    print("The patient is unlikely to have diabetes.")    