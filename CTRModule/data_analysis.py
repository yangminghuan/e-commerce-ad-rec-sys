from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

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

# 从hdfs中加载raw_sample.csv数据
schema = StructType([
    StructField("user", IntegerType()),
    StructField("time_stamp", LongType()),
    StructField("adgroup_id", IntegerType()),
    StructField("pid", StringType()),
    StructField("nonclk", IntegerType()),
    StructField("clk", IntegerType())
])
df = spark.read.csv("hdfs://localhost:9000/recsys/dataset/raw_sample.csv", header=True, schema=schema)

# 更改表结构，转换列名
new_df = df.withColumnRenamed("user", "userId").withColumnRenamed("time_stamp", "timestamp").\
    withColumnRenamed("adgroup_id", "adgroupId")

# 利用StringIndexer对指定字符串列进行类别编码处理
stringindexer = StringIndexer(inputCol='pid', outputCol='pid_feature')
# 进行OneHot独热编码
encoder = OneHotEncoder(dropLast=False, inputCol='pid_feature', outputCol='pid_value')
# 利用管道对数据进行OneHot编码处理
pipeline = Pipeline(stages=[stringindexer, encoder])
pipeline_model = pipeline.fit(new_df)
raw_sample_df = pipeline_model.transform(new_df)

# 加载ad_feature.csv数据
schema = StructType([
    StructField("adgroup_id", IntegerType()),
    StructField("cate_id", LongType()),
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

# 加载user_profile数据
schema = StructType([
    StructField("userId", IntegerType()),
    StructField("cms_segid", LongType()),
    StructField("cms_group_id", IntegerType()),
    StructField("final_gender_code", IntegerType()),
    StructField("age_level", IntegerType()),
    StructField("pvalue_level", StringType()),
    StructField("shopping_level", IntegerType()),
    StructField("occupation", IntegerType()),
    StructField("new_user_class_level", StringType())
])
df = spark.read.csv("hdfs://localhost:9000/recsys/dataset/user_profile.csv", header=True, schema=schema)
df = df.na.fill("-1")

# 对pvalue_level进行onehot编码
stringindexer = StringIndexer(inputCol="pvalue_level", outputCol="pl_onehot_feature")
encoder = OneHotEncoder(dropLast=False, inputCol="pl_onehot_feature", outputCol="pl_onehot_value")
pipeline = Pipeline(stages=[stringindexer, encoder])
pipeline_model = pipeline.fit(df)
new_df = pipeline_model.transform(df)

# 对new_user_class_level进行onehot编码
stringindexer = StringIndexer(inputCol="new_user_class_level", outputCol="nucl_onehot_feature")
encoder = OneHotEncoder(dropLast=False, inputCol="nucl_onehot_feature", outputCol="nucl_onehot_value")
pipeline = Pipeline(stages=[stringindexer, encoder])
pipeline_model = pipeline.fit(new_df)
user_profile_df = pipeline_model.transform(new_df)

# 进行数据合并
condition = [raw_sample_df.adgroupId == ad_feature_df.adgroupId]
_ = raw_sample_df.join(ad_feature_df, condition, 'outer')
condition = [_.userId == user_profile_df.userId]
dataset = _.join(user_profile_df, condition, 'outer')

# 剔除冗余的字段
useful_cols = ["timestamp", "clk", "pid_value", "price", "cms_segid", "cms_group_id", "final_gender_code", "age_level",
               "shopping_level", "occupation", "pl_onehot_value", "nucl_onehot_value"]
df_data = dataset.select(*useful_cols)
df_data = df_data.dropna()
# print(df_data.count())


# 根据特征字段计算特征向量
df_data = VectorAssembler().setInputCols(useful_cols[2:]).setOutputCol("features").transform(df_data)

# 根据时间划分数据集，将前7天数据作为训练集，最后一天作为测试集
train_data = df_data.filter(df_data.timestamp <= (1494691186-24*60*60))
test_data = df_data.filter(df_data.timestamp > (1494691186-24*60*60))

# 创建逻辑回归模型，进行模型训练并保存模型
lr = LogisticRegression()
model = lr.setLabelCol("clk").setFeaturesCol("features").fit(train_data)
model.save("hdfs://localhost:9000/recsys/models/CTRModel_Normal.obj")
