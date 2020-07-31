from bigdl.nn.criterion import MSECriterion
from zoo.pipeline.nnframes import *
from bigdl.util.common import init_engine, create_spark_conf
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from zoo.pipeline.nnframes import NNModel

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

cores = [1, 2, 3, 4]


conf = create_spark_conf() \
    .setAppName("Spark_Basic_Learning") \
    .setMaster("local[4]") \
    .set("spark.sql.warehouse.dir", "file:///C:/Spark/temp") \
    .set("spark.sql.streaming.checkpointLocation", "file:///C:/Spark/checkpoint") \
    .set("spark.sql.execution.arrow.enabled", "true")
    #.set("spark.sql.execution.arrow.maxRecordsPerBatch", "") # Utsav: Tweak only if memory limits are known. Default = 10,000

spark = SparkSession.builder \
    .config(conf=conf) \
    .getOrCreate()

# Init Big DL Engine
init_engine()

df = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZZ") \
    .load("../resources/datasets/dataset-1-converted.csv")



assembler = VectorAssembler(
    inputCols=["processing-time", "carparkID"],
    outputCol="features")


df = assembler.transform(df)


df = df.withColumnRenamed('slotOccupancy','label')

df = df.select('features','label')

try:
    bigdl_model = Model.load_keras(json_path="../resources/savedModels/keras_1.2.2/model.json", hdf5_path="../resources/savedModels/keras_1.2.2/weights.h5")

    print("Big Dl Model Created from keras .json and weights .h5 (pre-trained model ) ", bigdl_model)
except Exception as e:
    print(e)

criterion = MSECriterion()
estimator = NNEstimator(bigdl_model, criterion)

estimator.setMaxEpoch(50)\
         .setOptimMethod(Adam())\
         .setBatchSize(2048)

print("Before Training")

from datetime import datetime


print("4 Core")
print("Batches", 2048)

startTime = datetime.now()
# print("StartTime!\n", startTime)
for i in range(1000):
    trainedNN = estimator.fit(df)

endTime = datetime.now()
print("EndTime!\n", endTime)
print("Trained")

trainedNN.model.saveModel(modelPath="../resources/savedModel/bigdl/trainedNN.bigdl", weightPath="../resources/savedModel/bigdl/trainedNN.bin", over_write=True)

print("Saved!")





