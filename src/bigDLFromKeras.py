
from bigdl.util.common import init_engine, create_spark_conf
from bigdl.nn.layer import *
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


"""
OPTION 1: Use a .json file
"""
bigdl_model = Model.load_keras(json_path="../resources/savedModels/keras_1.2.2/model.json")

print("Big Dl Model Created from keras json ",bigdl_model)
"""
OPTION 2: Use a .h5 file (untrained)
"""
bigdl_model = Model.load_keras(hdf5_path="../resources/savedModels/keras_1.2.2/model.h5")
print("Big Dl Model Created from keras .h5 (untrained model ) ", bigdl_model)


"""
OPTION 3: Use a .h5 weights with .json (NB. Only useful for pre trained models)
"""
try:
    bigdl_model = Model.load_keras(json_path="../resources/savedModels/keras_1.2.2/model.json",hdf5_path="../resources/savedModels/keras_1.2.2/weights.h5")
    print("Big Dl Model Created from keras .json and weights .h5 (pre-trained model ) ", bigdl_model)
except Exception as e:
    print(e)


