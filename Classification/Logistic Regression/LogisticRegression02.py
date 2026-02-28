from sklearn.linear_model import LogisticRegression
X = [
    [1,50],
    [2,60],
    [3,65],
    [4,70],
    [5,80],
    [6,90]
]

y = [0,0,0,1,1,1]
model = LogisticRegression()
model.fit(X,y)
study_hours = int(input("Enter the studied hours:"))
attendace_level = float(input("Enter the atendance level : "))
predicted_result = model.predict([[study_hours, attendace_level]])[0]
if predicted_result == 1:
    print("Congratulations! You are likely to pass the exam.")
else:
    print("Don't worry! You can still improve and pass the exam with more effort.")    
