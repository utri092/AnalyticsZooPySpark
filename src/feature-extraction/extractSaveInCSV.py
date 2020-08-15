
from bigdl.util.common import init_engine, create_spark_conf

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
    .load("../../resources/datasets/dataset-1_converted.csv")


df.show()

df.printSchema()


from pyspark.sql.functions import  to_timestamp,date_format

# Convert processing time from int to timestamp
df = df.withColumn("processing-time",to_timestamp(col="processing-time"))


# Extract Day of Week and Week of Month using the feature extraction
# FIXME: This has been hardcoded to be put into carparkID & slotOccupancy cols need to create new cols instead

df = df.withColumn("carparkID",date_format(date ="processing-time",format= "EEEE"))

df.withColumn("slotOccupancy",date_format(date ="processing-time",format= "W")).show()