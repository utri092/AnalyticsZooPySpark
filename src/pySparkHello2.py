"""
Keras Sequential example

"""
# from bigdl.nn.layer import *
# from bigdl.optim.optimizer import *
# from bigdl.util.common import *

# create a graph model
# linear = Linear(10, 2)()
# sigmoid = Sigmoid()(linear)
# softmax = SoftMax()(sigmoid)
# model = Model([linear], [softmax])

# save it to Tensorflow model file
# model.save_tensorflow([("input", [4, 10])], "../resources/savedModels/model.pb")

import pandas as pd
import numpy as np
from zoo.pipeline.nnframes import NNModel

from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Dense
from bigdl.util.common import init_engine, create_spark_conf




conf = create_spark_conf() \
    .setAppName("Spark_Basic_Learning") \
    .setMaster("local[2]") \
    .set("spark.sql.warehouse.dir", "file:///C:/Spark/temp") \
    .set("spark.sql.streaming.checkpointLocation", "file:///C:/Spark/checkpoint") \
    .set("spark.sql.execution.arrow.enabled", "true")
    #.set("spark.sql.execution.arrow.maxRecordsPerBatch", "") # Utsav: Tweak only if memory limits are known. Default = 10,000

spark = SparkSession.builder \
    .config(conf=conf) \
    .getOrCreate()

# Init Big DL Engine
init_engine()

print("SparkSession start with BigDL!\n")

# TODO: Replicate in Spark and convert to Pd or ndArray using Arrow

df = pd.read_csv("../resources/datasets/dataset-1_converted.csv")

trainDf, testDf = train_test_split(df, test_size=0.2)
print("Created Train and Test Df\n")

predictionColumn = 'slotOccupancy'

x = trainDf.drop(columns=[predictionColumn])
inputs = len(x.columns)

y = trainDf[[predictionColumn]]
outputs = len(y.columns)

model = Sequential()
model.add(Dense(output_dim=inputs, activation="relu", input_shape=(inputs,)))
model.add(Dense(output_dim=inputs, activation="relu"))
model.add(Dense(output_dim=outputs))

model.compile(optimizer="adam", loss="mean_squared_error")

model.summary()
print("Created Sequential Model!\n")

xNumpy = x.to_numpy()
yNumpy = y.to_numpy()
# model.fit(x=xNumpy, y=yNumpy, nb_epoch=1, distributed=False)

import tensorflow as tf

weights = np.array(model.get_weights(), dtype=object)
print(weights)

tfModel = tf.keras.models.Sequential()

tfModel.add(tf.keras.layers.Dense(units=inputs, activation="relu", input_shape=(inputs,)))
tfModel.add(tf.keras.layers.Dense(units=inputs, activation="relu"))
tfModel.add(tf.keras.layers.Dense(units=inputs, activation="relu"))
tfModel.add(tf.keras.layers.Dense(units=outputs))

# tfModel.add(model.get_layer("Input"))
# tfModel.add(model.get_layer("Dense"))
# tfModel.add(model.get_layer("Dense"))
#
tfModel.compile(optimizer="adam", loss="mean_squared_error")
#
#
# print("Before set weights: {}\n".format(tfModel.get_weights()))
tfModel.summary()
# tfModel.set_weights(weights)
#
# tfModel.summary()

print("After set weights: {}\n".format(tfModel.get_weights()))
# nnModel = NNModel(model)
# nnModel.model.get_weights()

# weights = np.array(nnModel.model.get_weights(), dtype=object)

# # print(keras_model.get_weights())
# keras_model.save_model("../resources/savedModels/keras/lolModel.h5")
# # keras_model.save_weights(filepath="../resources/savedModels/keras/weights/lol.h5")