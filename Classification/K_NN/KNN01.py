from sklearn.neighbors import KNeighborsClassifier

X = [
    [180,7],
    [200, 7.5],
    [250 , 8],
    [300, 8.5],
    [330,9],
    [360,9.5]
]
# 0 =Apple , 1 = Orange
Y = [0,0,1,1,1,1]
knn  = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,Y)
weight = float(input("Enter the weight of the fruit in grams: "))
size = float(input("Enter the size of the fruit in cm: "))
prediction = knn.predict([[weight , size]])[0]
if prediction == 0:
    print("The fruit is an Apple.")
else:
    print("The fruit is an Orange.")