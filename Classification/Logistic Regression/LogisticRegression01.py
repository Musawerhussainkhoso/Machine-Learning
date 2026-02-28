from sklearn.linear_model import LogisticRegression

#create obj
model = LogisticRegression()
X = [[1],[2],[3],[4],[5]]
Y= [0,0,0,1,1]#0 consider fail , 1 consider pass
model.fit(X,Y)
hours = float(input("Enter the studied hours:"))
predicted_result = model.predict([[hours]])[0]
#We use [0] because predict() returns a list (array), and we need the first value from it.

if predicted_result == 1:
    print("Congratulations! You are likely to pass the exam.")
else:
    print("Don't worry! You can still improve and pass the exam with more effort.")
