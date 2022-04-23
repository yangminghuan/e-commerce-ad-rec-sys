from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import redis


def recall_cate_by_cf(partition):
    # 建立redis连接池和客户端
    pool = redis.ConnectionPool(host='localhost', port=6379, password="123456")
    client = redis.Redis(connection_pool=pool)

    for row in partition:
        client.hset("recall_cate", row.userId, str([i.cateId for i in row.recommendations]))


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

# 从hdfs中加载模型
als_model = ALSModel.load("hdfs://localhost:9000/recsys/models/userCateRatingALSModel.obj")

# 召回用户喜欢的top-5个类别
recall_result = als_model.recommendForAllUsers(5)
recall_result.show(10)

# 将模型召回结果存储到redis中
recall_result.foreachPartition(recall_cate_by_cf)
print(recall_result.count())
