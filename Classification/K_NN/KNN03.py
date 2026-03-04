import pandas as pn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score ,confusion_matrix

read_file = pn.read_excel("Dataset.xlsx")
copy_file = read_file.copy()
print(copy_file.head())
print(copy_file.isnull().sum())
copy_file.dropna(inplace=True)
print(copy_file.isnull().sum())

X = copy_file.drop("Pass", axis=1)
y = copy_file["Pass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
