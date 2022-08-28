import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("Fertilizer_prediction.csv")

inputs = data.drop('Fertilizer', axis="columns")
target = data['Fertilizer']

crop_type_label_encoder = LabelEncoder()
croptype_dict = {}
data["Crop "] = crop_type_label_encoder.fit_transform(data["Crop "])
for i in range(len(data["Crop "].unique())):
    croptype_dict[i] = crop_type_label_encoder.inverse_transform([i])[0]

print(croptype_dict)
soil_type_label_encoder = LabelEncoder()
data["Soil "] = soil_type_label_encoder.fit_transform(data["Soil "])
soiltype_dict = {}
for i in range(len(data["Soil "].unique())):
    soiltype_dict[i] = soil_type_label_encoder.inverse_transform([i])[0]
print(soiltype_dict)


le_soil = LabelEncoder()
le_crop = LabelEncoder()
inputs['Soil '] = le_soil.fit_transform(inputs['Soil '])
inputs['Crop '] = le_crop.fit_transform(inputs['Crop '])


x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=0)

model1 = GaussianNB()
model1.fit(x_train.values, y_train)
pickle.dump(model1, open('model1.pkl', 'wb'))
pickle.dump(croptype_dict, open("croptype_dict.pkl", "wb"))
pickle.dump(soiltype_dict, open("soiltype_dict.pkl", "wb"))

