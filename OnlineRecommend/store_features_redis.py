from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
import json
import redis


def store_ad_redis(partition):
    client = redis.StrictRedis(host="localhost", port=6379, password="123456", db=10)
    for r in partition:
        data = {"price": r.price}
        # 转为json字符串保存
        client.hsetnx("ad_features", r.adgroupId, json.dumps(data))


def store_user_redis(partition):
    client = redis.StrictRedis(host="localhost", port=6379, password="123456", db=10)
    for r in partition:
        data = {
            "cms_group_id": r.cms_group_id,
            "final_gender_code": r.final_gender_code,
            "age_level": r.age_level,
            "shopping_level": r.shopping_level,
            "occupation": r.occupation,
            "pvalue_level": r.pvalue_level,
            "new_user_class_level": r.new_user_class_level
        }
        # 转为json字符串保存
        client.hsetnx("user_features", r.userId, json.dumps(data))


SPARK_APP_NAME = "ALSRecommend"
SPARK_URL = "spark://ymh:7077"

config = (
    ("spark.app.name", SPARK_APP_NAME),
    ("spark.executor.memory", "10g"),
    ("spark.master", SPARK_URL),
    ("spark.executor.cores", "3"),
)
conf = SparkConf()
conf.setAll(config)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# 先将广告ad和用户特征缓存到redis中
feature_cols_from_ad = ["price"]
feature_cols_from_user = ["cms_group_id", "final_gender_code", "age_level", "shopping_level", "occupation",
                          "pvalue_level", "new_user_class_level"]

# 加载ad_feature.csv数据
schema = StructType([
    StructField("adgroup_id", IntegerType()),
    StructField("cate_id", IntegerType()),
    StructField("campaign_id", IntegerType()),
    StructField("customer", IntegerType()),
    StructField("brand", IntegerType()),
    StructField("price", FloatType())
])
df = spark.read.csv("hdfs://localhost:9000/recsys/dataset/ad_feature.csv", header=True, schema=schema)
df = df.replace("NULL", "-1")
ad_feature_df = df.withColumnRenamed("adgroup_id", "adgroupId").withColumnRenamed("cate_id", "cateId").\
    withColumnRenamed("campaign_id", "campaignId").withColumnRenamed("customer", "customerId").\
    withColumnRenamed("brand", "brandId")
# 缓存到redis中
ad_feature_df.foreachPartition(store_ad_redis)

# 加载user_profile数据
schema = StructType([
    StructField("userId", IntegerType()),
    StructField("cms_segid", IntegerType()),
    StructField("cms_group_id", IntegerType()),
    StructField("final_gender_code", IntegerType()),
    StructField("age_level", IntegerType()),
    StructField("pvalue_level", IntegerType()),
    StructField("shopping_level", IntegerType()),
    StructField("occupation", IntegerType()),
    StructField("new_user_class_level", IntegerType())
])
user_profile_df = spark.read.csv("hdfs://localhost:9000/recsys/dataset/user_profile.csv", header=True, schema=schema)
# 缓存到redis中
user_profile_df.foreachPartition(store_user_redis)
