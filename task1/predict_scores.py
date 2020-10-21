import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


data = pd.read_csv("scores.csv")
X = np.array(data['Hours']).reshape(-1,1)
y = data['Scores']
reg = LinearRegression()
reg.fit(X,y)
h = int(input("Enter hours of study\n"))
p = reg.predict(np.array(h).reshape(-1,1))

print(f"Predicted score for {h} hours of study is : {p[0]}")
