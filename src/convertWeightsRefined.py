import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

############### STEP 1: Create Keras Model with Same Architecure as BigDL Model ############
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

kerasModelEmptyWeights = model.get_weights()
print("Model1 Weights before setting from Cluster! \n")
print(kerasModelEmptyWeights)


############### STEP 2: Load Trained BigDL Model ###############

from bigdl.util.common import init_engine, create_spark_conf
from pyspark.sql import SparkSession
from bigdl.nn.layer import *

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

init_engine()

bigDlModel = Model.loadModel(modelPath="../resources/savedModels/bigdl/trainedNN.bigdl", weightPath="../resources/savedModels/bigdl/trainedNN.bin")

print("Loaded BigDL Model !")

zooWeights = bigDlModel.get_weights()

############### STEP 3:  Reshape Weights & Bias Arrays of BigDL Model to Keras Model's Original Shape

# Note:- If reshape errors occur. That means architecure(model.summary()) has not been replicated exactly
for i in range(len(zooWeights)):

    zooWeights[i] = zooWeights[i].reshape(kerasModelEmptyWeights[i].shape)

model.set_weights(zooWeights)

model.save(filepath="../resources/savedModels/keras_1.2.2/convertedInferenceModel.h5")

print("Saved Model!")