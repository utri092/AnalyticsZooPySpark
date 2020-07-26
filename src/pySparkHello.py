"""
TfPark Sequential example

"""
import tensorflow as tf
import pandas as pd
from zoo.common.nncontext import *
from zoo.pipeline.nnframes import NNModel
from zoo.tfpark import TFDataset
from pyspark.sql import SparkSession
from zoo.tfpark import KerasModel, TFDataset
from sklearn.model_selection import train_test_split
from bigdl.nn.criterion import *
from zoo.pipeline.nnframes import NNEstimator
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

# Init Big DL Engine
init_engine()

print("SparkSession start with BigDL!\n")

# TODO: Replicate in Spark and convert to Pd or ndArray using Arrow

# df = spark.read.format("csv") \
#     .option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZZ") \
#     .option("inferSchema", "true") \
#     .option("header", "true") \
#     .load("../resources/datasets/dataset_1_converted.csv")

# df.show()

df = pd.read_csv("../resources/datasets/dataset_1_converted.csv")

# trainDf, testDf = df.randomSplit([0.8, 0.2])

trainDf, testDf = train_test_split(df, test_size=0.2)
print("Created Train and Test Df\n")

predictionColumn = 'slotOccupancy'

x = trainDf.drop(columns=[predictionColumn])
inputDim = len(x.columns)

y = trainDf[[predictionColumn]]
outputDim = len(y.columns)

# training_dataset = TFDataset.from_dataframe(df=trainDf, labels_cols=["slotOccupancy"], feature_cols=["carparkID", "processing-time"], shuffle=False, batch_size=32)
# training_dataset = TFDataset.from_ndarrays(tensors=x.values,batch_size=32)
# print("Created TF Dataset\n")

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(inputDim, activation="relu", input_shape=(2,)),
     tf.keras.layers.Dense(inputDim, activation='relu'),
     tf.keras.layers.Dense(outputDim),
     ]
)

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              )

keras_model = KerasModel(model)
print("Created Keras Model! \n")

# print("batchSize TFDataset: {}".format(training_dataset.batch_size))
# keras_model.fit(x=x.values, y=y.values, epochs=5)
print("Training Complete!\n")
# keras_model.save_model("../resources/savedModels/tfParkModel.h5")

weights = keras_model.get_weights()
# weights = np.array(weights, dtype=object)
# print(weights, type(weights))

kModel = Model()

# keras_model.save_weights("../resources/savedModels/keras/weights/wt.h5")

# keras_model.save_model("../resources/savedModels/keras/model.h5")
