from sklearn.linear_model import LinearRegression
#A company wants to predict salary based on years of experience.
x = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y = [30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
model = LinearRegression()
model.fit(x,y)
predicted_salary = int(input("Enter years of experience:"))
predicted_salary = model.predict([[predicted_salary]])[0]
print("Predicted Salary:", predicted_salary)