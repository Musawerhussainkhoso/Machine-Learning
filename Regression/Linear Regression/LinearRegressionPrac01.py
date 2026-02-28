from sklearn.linear_model import LinearRegression

lm = LinearRegression()
X = [[1],[2]]
Y = [34,55]
lm.fit(X,Y)
pred_value = int(input("Enter no of hours:"))
print(lm.predict([[pred_value]]))
