import pandas as pn 
from sklearn.preprocessing import LabelEncoder
read_data = pn.read_excel('emp_data.xlsx')
copy_file = read_data.copy()
print(copy_file.isnull().sum())
print(copy_file.head())
#object for label encoding
model = LabelEncoder()
copy_file['Gender_Encoded'] = model.fit_transform(copy_file['Gender'])
print('\n Label Encoded DataFrame:')
print(copy_file['Gender_Encoded'])
copy_file = pn.get_dummies(copy_file , columns=['Education','Department'])
print('\n Hot Encoded DataFrame:')
print(copy_file.filter(like='Education'))
print(copy_file.filter(like='Department'))