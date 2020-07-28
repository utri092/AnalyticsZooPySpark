"""
Generates a simple nn model using keras 1.2.2 and stores it in a model - See 2 formats Listed below

"""

import pandas as pd
from sklearn.model_selection import train_test_split
import keras

"""
TODO: Deprecated: Will be refactored to read data when training a model
# Read the dataset 1
df = pd.read_csv("../resources/datasets/dataset-1-converted.csv")

# Split into training and test data
trainDf, testDf = train_test_split(df, test_size=0.2)
print("Created Train and Test Df\n")

predictionColumn = 'slotOccupancy'

x = trainDf.drop(columns=[predictionColumn])
inputs = len(x.columns)

y = trainDf[[predictionColumn]]
outputs = len(y.columns)


"""



inputs = 2
outputs =1

model = keras.models.Sequential()
model.add(keras.layers.Dense(inputs, activation='relu', input_shape=(inputs,)))
model.add(keras.layers.Dense(inputs, activation='relu'))
model.add(keras.layers.Dense(outputs))

log_dir = "../resources/board/model_log"
app_name = "zooKeras"

print(keras.__version__)

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

print("Before Save\n")
"""
Option 1 : Save architecture as .json and weights as .h5
"""
model_json = model.to_json()
with open("../resources/savedModels/keras_1.2.2/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../resources/savedModels/keras_1.2.2/weights.h5")
"""
Option 2 : Save entire model as .h5
"""

model.save("../resources/savedModels/keras_1.2.2/model.h5")

print("Saved!\n")