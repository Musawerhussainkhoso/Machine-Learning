from sklearn.preprocessing import LabelEncoder
import pandas as pb 
read_file = pb.read_excel('categorical_dataset.xlsx')
df_label = read_file.copy()

#create object 
le = LabelEncoder()
df_label['Gender_Encoded'] = le.fit_transform(df_label['Gender']) 
#df_label['City_Encoded'] = le.fit_transform(df_label['City'])

print("\n Label Encoded DataFrame:")
print(df_label[['Name','Gender','Gender_Encoded','Department']])

'''df_hot_encoded = pb.get_dummies(df_label, columns=['City'])
print("\n One-Hot Encoded DataFrame:")
print(df_hot_encoded)'''

