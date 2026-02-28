import pandas as pn 
from sklearn.preprocessing import LabelEncoder
read_data = pn.read_excel('categorical_dataset.xlsx')
copy_file = read_data.copy()
#make dataframe 
df = pn.DataFrame(copy_file)
#create object
le = LabelEncoder()
print(df.head())
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
print('\n Encoded DataFrame:')
print(df[['Name','Gender','Gender_Encoded','City','Department']])
df= pn.get_dummies(df , columns=['City'])
df= pn.get_dummies(df, columns = ['Department'])
print('\n Hot Encoded DataFrame By City')
print(df.filter(like='City'))
print('\n Hot Encoded DataFrame By Department:')
print(df.filter(like = 'Department'))
