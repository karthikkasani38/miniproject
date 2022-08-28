import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("Crop_recommendation.csv")

inputs = data.drop('label', axis="columns")
target = data['label']

x_train, x_test, y_train, y_test = train_test_split(inputs, target,test_size=0.33)



model= GaussianNB()
model.fit(x_train, y_train)

pickle.dump(model, open('model.pkl','wb'))

