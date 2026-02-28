import pandas as pb 
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split

data = {
    "StudyHours" : [1,2,3,4,5],
    "TestScore" : [50,60,70,80,90]
}

df = pb.DataFrame(data)
#Standard Scaler
Scaler = StandardScaler()
scaled_data = Scaler.fit_transform(df)

print("Standard Scaler Output:")
print(pb.DataFrame(scaled_data, columns= ["StudyHours","TestScore"]))

MinMax = MinMaxScaler()
minmax_scaled_data = MinMax.fit_transform(df)
print("\nMin-Max Scaler Output:")
print(pb.DataFrame(minmax_scaled_data, columns= ["StudyHours","TestScore"]))

