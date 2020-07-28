import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("../resources/datasets/dataset_1_converted.csv")

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
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = tf.keras.Model(inputs=parkingInput, outputs=lastLayer, name="functionalModel")

model.compile(optimizer='adam', loss='mean_squared_error')

# model.summary()

print(model.get_weights())
weights = model.get_weights()
print("Model1 Weights\n")
print(weights)

model.save_weights("../resources/savedModels/keras/weights/wt.h5")

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

parkingInput2 = Input(shape=(inputs,))
print(parkingInput2.shape)

denseLayer2 = Dense(output_dim=inputs, activation="relu")
hidden2 = denseLayer2(parkingInput2)

lastLayer2 = Dense(output_dim=outputs,activation="relu")(hidden2)
model2 = Model(input=parkingInput2, output=lastLayer2, name="functionalModel2")
# model2 = Model(inputs=[parkingInput2], outputs=[lastLayer2])

log_dir = "../resources/board/model_log"
app_name = "zooKeras"
model2.set_tensorboard(log_dir = log_dir, app_name=app_name)

model2.compile(optimizer='adam', loss='mean_squared_error')

model2.fit(x=x.to_numpy(), y=y.to_numpy(), nb_epoch=2, distributed=False)
model2.summary()

weights2 = model2.get_weights()

layers = model2.layers

layersList = []

for i, layer in enumerate(layers):

    try:

        params = layer.parameters()

        layerName = list(params.keys())

        # # Get params
        # if len(layerName)> 0:

        layerAttr = params[layerName[0]]

        wt = layerAttr['weight']

        bias = layerAttr['bias']
        # Hardcoded - Transpose Last Layer
        if i==2:

            wtT = np.empty((2,1))

            wt = wt.reshape(2, 1)
            print(wt)

        layersList.append(wt)
        layersList.append(bias)



    except Exception as e:
        print(e)

        pass


print("done")


model.set_weights(layersList)

model.save_weights(filepath="../resources/savedModels/lol.h5")
model.save(filepath="../resources/savedModels/lol2.h5")