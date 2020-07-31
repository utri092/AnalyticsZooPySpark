import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from zoo.pipeline.nnframes import NNModel

df = pd.read_csv("../resources/datasets/dataset-1_converted.csv")

trainDf, testDf = train_test_split(df, test_size=0.2)
print("Created Train and Test Df\n")

predictionColumn = 'slotOccupancy'

x = trainDf.drop(columns=[predictionColumn])
inputs = len(x.columns)

y = trainDf[[predictionColumn]]
outputs = len(y.columns)

parkingInput = tf.keras.Input(shape=(inputs,))
print(parkingInput.shape)

# Hidden Layer/s
denseLayer = tf.keras.layers.Dense(units=inputs,activation="relu")
hidden = denseLayer(parkingInput)
lastLayer = tf.keras.layers.Dense(units=outputs,activation="relu")(hidden)

log_dir = "../resources/board/model_log"
app_name = "zooKeras"

model = tf.keras.Model(inputs=parkingInput, outputs=lastLayer, name="functionalModel")

model.compile(optimizer='adam', loss='mean_squared_error')

inferenceWeights = model.get_weights()
print("Model1 Weights before setting from Cluster! \n")
print(inferenceWeights)

from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import Input, Dense
from bigdl.util.common import init_engine, create_spark_conf
from pyspark.sql import SparkSession

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

# parkingInput2 = Input(shape=(inputs,))
# print(parkingInput2.shape)
#
# denseLayer2 = Dense(output_dim=inputs, activation="relu")
# hidden2 = denseLayer2(parkingInput2)
#
# lastLayer2 = Dense(output_dim=outputs,activation="relu")(hidden2)
# model2 = Model(input=parkingInput2, output=lastLayer2, name="functionalModel2")
#
#
# model2.compile(optimizer='adam', loss='mean_squared_error')
#
# # Set Tensorboard
# log_dir = "../resources/board/model_log"
# app_name = "zooKeras"
# model2.set_tensorboard(log_dir = log_dir, app_name=app_name)
#
# model2.fit(x=x.to_numpy(), y=y.to_numpy(), nb_epoch=2, distributed=False)
# model2.summary()

from tensorflow.keras.models import load_model
model2 = load_model("../resources/savedModels/bigdl/model.h5")

# nnModel = NNModel(model2)

# zooWeights = nnModel.model.get_weights()
# zooWeights = model2.get_weights()
#
# for i in range(len(zooWeights)):
#
#     zooWeights[i] = zooWeights[i].reshape(inferenceWeights[i].shape)
#
# model.set_weights(zooWeights)
#
# model.save(filepath="../resources/savedModels/keras/inferenceModel.h5")

# print("Saved Model!")