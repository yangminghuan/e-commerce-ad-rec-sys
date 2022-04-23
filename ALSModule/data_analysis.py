from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

SPARK_APP_NAME = "ALSRecommend"
SPARK_URL = "spark://ymh:7077"

config = (
    ("spark.app.name", SPARK_APP_NAME),
    ("spark.executor.memory", "6g"),
    ("spark.master", SPARK_URL),
    ("spark.executor.cores", "3"),
)
conf = SparkConf()
conf.setAll(config)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# 构建结构对象
schema = StructType([
    StructField("userId", IntegerType()),
    StructField("timestamp", LongType()),
    StructField("btag", StringType()),
    StructField("cateId", IntegerType()),
    StructField("brandId", IntegerType())
])

# 从hdfs加载数据为sparkDataFrame，并设置结构
behavior_log_df = spark.read.csv("hdfs://localhost:9000/recsys/dataset/behavior_log.csv", header=True, schema=schema)

# 统计每个用户对各个商品类别cate和品牌brand的pv、fav、cart、buy数量
cate_count_df = behavior_log_df.groupBy(behavior_log_df.userId, behavior_log_df.cateId).pivot('btag',
                                        ['pv', 'fav', 'cart', 'buy']).count()
brand_count_df = behavior_log_df.groupBy(behavior_log_df.userId, behavior_log_df.brandId).pivot('btag',
                                         ['pv', 'fav', 'cart', 'buy']).count()

# 将数据存储到hdfs中，便于后续其他操作使用
cate_count_df.write.csv("hdfs://localhost:9000/recsys/preprocessing_dataset/cate_count.csv", header=True)
brand_count_df.write.csv("hdfs://localhost:9000/recsys/preprocessing_dataset/brand_count.csv", header=True)
