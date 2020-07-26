"""
Extract/Write Weights from Model

"""


import h5py
import numpy as np
import tensorflow as tf


keras_model = tf.keras.models.load_model("../resources/savedModels/keras/model.h5")

weights = keras_model.get_weights()
weights = np.array(weights, dtype=object)
# print(weights)

# keras_model.save_weights("../resources/savedModels/keras/weights/wt3.txt")

def func(name, obj):

    # if isinstance(obj, h5py.Group):
    #     print(obj)

    if isinstance(obj, h5py.Dataset):

        print(obj.values)

f = h5py.File("../resources/savedModels/keras/weights/wt.h5", 'r')

print(f.keys())
f.visititems(func)
