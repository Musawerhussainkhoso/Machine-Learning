import pandas as pn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pn.read_excel('DatasetForSplit.xlsx')
print(data.head())

x = data.drop('target', axis=1)
y = data['target']

x_train , x_test , y_train , y_test = train_test_split ( x , y , test_size = 0.2 , random_state= 42)
ss = StandardScaler()
x_train_scaled = ss.fit_transform(x_train)
x_test_scaled = ss.fit_transform(x_test)

# Now X_train_scaled and X_test_scaled are ready for ML models
print("Training features shape:", x_train_scaled.shape)
print("Testing features shape:", x_test_scaled.shape)