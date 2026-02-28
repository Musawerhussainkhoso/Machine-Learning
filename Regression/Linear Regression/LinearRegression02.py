from sklearn.linear_model import LinearRegression
# Independent variable (X)
x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

# Dependent variable (Y)
y = [15, 20, 25, 35, 40, 50, 55, 65, 70, 80]
model = LinearRegression()
model.fit(x,y)
hours = int(input("Enter the hours :"))
predicted_marks = model.predict([[hours]])[0]
print("Predicted Marks:", predicted_marks)
