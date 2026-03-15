import pandas as pn 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
read_data = pn.read_csv('insurance.csv')
#Step 01: EDA
print(read_data.head())
print(read_data.tail())
print(read_data.info())
print(read_data.describe())
print(read_data.isnull().sum())

#visualization
#select numeric columns 
print(read_data.columns)
numeric_columns = ['age', 'bmi', 'children', 'charges']
for column in numeric_columns:
    plt.figure(figsize=(8, 4))#creating the fig 8 to 4 pixels 
    sns.histplot(read_data[column], kde=True)#creating the histogram with 
    #the kde line, kde is the kernel density estimation which is a 
    # way to estimate the probability density function of a random 
    # variable
    plt.title(f'Distribution of {column}')#setting the title of the plot with the column name
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=read_data[col])#creating the box plot for each numeric column
        plt.title(f'Box Plot of {col}')#setting the title of the plot with the column name
        plt.xlabel(col)
        plt.show()
        plt.figure(figsize=(8, 4))
        sns.heatmap(read_data[numeric_columns].corr(), annot=True, cmap='coolwarm')#creating the heatmap for the correlation between the numeric columns
        plt.title('Correlation Heatmap')#setting the title of the plot
        plt.show()
#data cleaaning 

        