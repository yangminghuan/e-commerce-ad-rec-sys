from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.ml.recommendation import ALS

SPARK_APP_NAME = "ALSRecommend"
SPARK_URL = "spark://ymh:7077"


# 评分函数
def process_row(r):
    pv_count = r.pv if r.pv else 0.0
    fav_count = r.fav if r.fav else 0.0
    cart_count = r.cart if r.cart else 0.0
    buy_count = r.buy if r.buy else 0.0

    pv_score = 0.2 * pv_count if pv_count <= 20 else 4.0
    fav_score = 0.4 * fav_count if fav_count <= 20 else 8.0
    cart_score = 0.6 * cart_count if cart_count <= 20 else 12.0
    buy_score = 1 * buy_count if buy_count <= 20 else 20.0

    rating = pv_score + fav_score + cart_score + buy_score
    return r.userId, r.brandId, rating


config = (
    ("spark.app.name", SPARK_APP_NAME),
    ("spark.executor.memory", "10g"),
    ("spark.master", SPARK_URL),
    ("spark.executor.cores", "3"),
)
conf = SparkConf()
conf.setAll(config)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# 设置检查点，防止模型训练造成内存溢出
spark.sparkContext.setCheckpointDir("hdfs://localhost:9000/recsys/checkPoint/")

# 构建结构对象
schema = StructType([
    StructField("userId", IntegerType()),
    StructField("brandId", IntegerType()),
    StructField("pv", IntegerType()),
    StructField("fav", IntegerType()),
    StructField("cart", IntegerType()),
    StructField("buy", IntegerType())
])

# 从hdfs加载数据
brand_count_df = spark.read.csv("hdfs://localhost:9000/recsys/preprocessing_dataset/brand_count.csv", header=True,
                               schema=schema)
# 用户对商品类别进行评分
brand_rating_df = brand_count_df.rdd.map(process_row).toDF(["userId", "brandId", "rating"])
brand_rating_df.show(10)

# 利用打分数据，训练ALS模型
als = ALS(userCol='userId', itemCol='brandId', ratingCol='rating', checkpointInterval=5)
model = als.fit(brand_rating_df)

# 将训练完成的模型进行存储
model.save("hdfs://localhost:9000/recsys/models/userBrandRatingALSModel.obj")

# # 利用训练好的模型给用户推荐top-n个类别物品
# ret = model.recommendForAllUsers(3)
# ret.show(10)
