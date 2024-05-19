import numpy as np
import math
import pandas as pd
from sklearn import linear_model
from word2number import w2n

df=pd.read_csv('hiring.csv')
df.test_score=df.test_score.fillna(math.floor(df.test_score.median()))
df.experience = [0 if pd.isna(x) else w2n.word_to_num(x) for x in df.experience]
reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score','interview_score']],df.salary)
print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))
