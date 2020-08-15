import keras
print(keras.__version__)

inputs = 9
outputs = 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(inputs, activation='relu', input_shape=(inputs,)))
model.add(keras.layers.Dense(inputs, activation='relu'))
model.add(keras.layers.Dense(inputs, activation='relu'))
model.add(keras.layers.Dense(inputs, activation='relu'))
model.add(keras.layers.Dense(outputs))

log_dir = "../resources/board/model_log"
app_name = "zooKeras"

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

print("Before Save\n")
"""
Option 1 : Save architecture as .json and weights as .h5
"""
model_json = model.to_json()
with open("../../resources/newModels/Keras_1.2.2/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../../resources/newModels/Keras_1.2.2/weights.h5")
"""
Option 2 : Save entire model as .h5
"""

model.save("../../resources/newModels/Keras_1.2.2/model.h5")

print("Saved!\n")