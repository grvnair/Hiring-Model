import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("H:/Datasets/Python Projects/Hiring/hiring.csv")

df['experience'].fillna(0, inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)

# converting the strings in experience columns to integers

def word_to_int(word):
    word_dict = {'one' : 1, 'two' : 2, 'three' : 3, 'four' : 4, 'five' : 5, 'six' : 6, 'seven' : 7, 'eight' : 8, 'nine' : 9,
                 'ten' : 10, 'eleven' : 11, 0:0}
    return word_dict[word]

df['experience'] = df['experience'].apply(lambda k : word_to_int(k))

X = df.drop('salary', axis='columns')
y = df['salary']

regressor = LinearRegression()
regressor.fit(X, y)

# saving the model
pickle.dump(regressor, open('hiring_model.pkl', 'wb'))

# loading the model
model = pickle.load(open('hiring_model.pkl', 'rb'))