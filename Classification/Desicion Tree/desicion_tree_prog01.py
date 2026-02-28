from sklearn.tree import DecisionTreeClassifier

X = [
    [7,2],#Apple
    [8 ,3],#Apple
    [9,8],#Orange
    [10,9]#Orange
]
y = [0,0,1,1] # 0 for apple , 1 for oranges
model = DecisionTreeClassifier()
model.fit(X,y)

fruit_size = float(input(" Enter the size of the fruit: "))
fruit_col =  float(input("Enter the color shade (1-0):"))
result = model.predict([[fruit_size , fruit_col]])[0]
if result == 0:
    print("The fruit is an Apple.")
else:
    print("The fruit is an orange")    
