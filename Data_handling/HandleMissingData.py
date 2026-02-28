import pandas as pb 
data = {
    "Name": ["Ali", "Sara", "Ahmed", "Ayesha", "Bilal", "Hina"],
    "Age": [20, 22, None, 19, 23, None],
    "Marks": [85, None, 78, 92, None, 88],
    "City": ["Karachi", None, "Lahore", "Karachi", "Islamabad", None]
}
df = pb.DataFrame(data)
print(df)
print(df.isnull().sum())
df_drop = df.dropna()
print(df_drop)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Marks'].fillna(df['Marks'].mean(), inplace=True)
df['City'].fillna(df['City'].mode()[0], inplace=True)
print(df)