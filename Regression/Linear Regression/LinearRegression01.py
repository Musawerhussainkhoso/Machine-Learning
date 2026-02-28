from sklearn.linear_model import LinearRegression

#input
X = [[1],[2],[3],[4],[5]]
Y = [40,50,65,75,90]
model = LinearRegression()
model.fit(X,Y)
#model.predict([[6]])
hours = float(input("How many hours you studied ? "))
predicted_value = model.predict([[hours]])
print("predicted result:", predicted_value)