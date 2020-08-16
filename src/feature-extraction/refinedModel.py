
from bigdl.util.common import init_engine, create_spark_conf


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

df.printSchema()
from pyspark.sql.functions import  to_timestamp,date_format
from pyspark.sql.functions import year,month,dayofmonth, dayofweek, hour, minute, second


from pyspark.sql import column

# Convert processing time from int to timestamp
df = df.withColumn("processing-time",to_timestamp(col="processing-time"))

# Add columns for extracted features

print("After Conversion \n")
df.printSchema()

# Extract Day of Week and Week of Month using the feature extraction

df = df.withColumn("dayOfWeek",dayofweek(col="processing-time"))\
    .withColumn("year",year(col="processing-time"))\
    .withColumn("month",month(col="processing-time"))\
    .withColumn("hour",hour(col="processing-time"))\
    .withColumn("minute",minute(col="processing-time"))\
    .withColumn("second",second(col="processing-time"))\
    .drop('processing-time')

df = df.withColumn("timeOfDay", col=(df['hour']*100+ df['minute']))\
    .drop('hour', 'minute', 'second')


df.show()

df.toPandas().to_csv(path_or_buf="../../resources/newDatasets/dataset-1_refined.csv")



