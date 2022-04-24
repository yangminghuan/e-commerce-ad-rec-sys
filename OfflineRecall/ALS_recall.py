from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, FloatType
from pyspark.ml.recommendation import ALSModel
import redis
import pandas as pd
import numpy as np

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
ad_cate_df = ad_feature_df.select("adgroupId", "cateId").toPandas()

# 利用基于商品类别的ALS模型进行召回
# 加载ALS模型
als_model = ALSModel.load("hdfs://localhost:9000/recsys/models/userCateRatingALSModel.obj")

# 存储用户召回的商品id，使用redis第9号数据库，类型：sets类型
client = redis.StrictRedis(host="localhost", port=6379, password="123456", db=9)

# 循环遍历每一个用户，召回商品id并存储
for r in als_model.userFactors.select("id").collect():
    userId = r.id
    cateId_df = pd.DataFrame(ad_cate_df["cateId"].unique(), columns=["cateId"])
    cateId_df.insert(0, "userId", np.array([userId for i in range(6769)]))
    ret = set()
    # 利用ALS模型求用户对所有商品类别的兴趣程度
    cateId_list = als_model.transform(spark.createDataFrame(cateId_df)).sort("prediction", ascending=False).na.drop()
    # 从前10个分类中选出500个商品进行召回
    for i in cateId_list.head(10):
        need = 500 - len(ret)
        ret = ret.union(np.random.choice(ad_cate_df[ad_cate_df['cateId'] == i.cateId].adgroupId.dropna().\
                                         astype(np.string_), need))
        if len(ret) > 500:
            break
    # 缓存到redis中
    client.sadd(userId, *ret)

