from bigdl.nn.criterion import MSECriterion
from zoo.pipeline.nnframes import *
from bigdl.util.common import init_engine, create_spark_conf
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from zoo.pipeline.nnframes import NNModel

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

conf = create_spark_conf() \
    .setAppName("Spark_Basic_Learning") \
    .setMaster("local[4]") \
    .set("spark.sql.warehouse.dir", "file:///C:/Spark/temp") \
    .set("spark.sql.streaming.checkpointLocation", "file:///C:/Spark/checkpoint") \
    .set("spark.sql.execution.arrow.enabled", "true")\
    # .set('spark.sql.execution.arrow.fallback.enabled', 'true')
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
    .load("../resources/datasets/dataset-1_converted.csv")

trainDf = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZZ") \
    .load("../resources/datasets/training-dataset1.csv")

testDf = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZZ") \
    .load("../resources/datasets/test-dataset1.csv")



trainedNN = Model.loadModel(modelPath="../resources/savedModel/bigdl/trainedNN.bigdl", weightPath="../resources/savedModel/bigdl/trainedNN.bin")
print("Loaded Trained NN!\n", trainedNN)

testDf = testDf.select('processing-time', 'carparkID')

testDf.show(n=20)
print("Test DF!\n")
pdTestNumpy = testDf.toPandas().to_numpy()
print(pdTestNumpy.shape)
# FIXME: Bad Predictions
predictions =  trainedNN.predict(pdTestNumpy)
print(predictions)
print("Predictions!\n")