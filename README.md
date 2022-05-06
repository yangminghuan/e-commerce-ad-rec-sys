# e-commerce-ad-rec-sys
## 电商广告推荐系统
该项目数据基于阿里巴巴提供的一个淘宝展示广告点击率预估数据集Ali_Display_Ad_Click.  
数据集下载地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=56  
### 数据集介绍
- raw_sample.csv：随机抽取的114万用户8天内的广告展示/点击日志（2600万条记录），字段说明如下：  
(1) user_id：脱敏过的用户ID；  
(2) adgroup_id：脱敏过的广告单元ID；  
(3) time_stamp：时间戳；  
(4) pid：资源展示位置；  
(5) noclk：没有被点击；  
(6) clk：被点击

- ad_feature.csv：涵盖了raw_sample.csv数据中的全部广告信息（约80万条），字段说明如下：  
(1) adgroup_id：脱敏过的广告单元ID；  
(2) cate_id：脱敏过的商品类别ID；  
(3) campaign_id：脱敏过的广告计划ID；  
(4) customer_id：脱敏过的广告主ID；  
(5) brand_id：脱敏过的品牌ID；  
(6) price：商品的价格

- user_profile.csv：涵盖了raw_sample.csv数据中的全部用户的基本信息（约100多万用户），字段说明如下：  
(1) user_id：脱敏过的用户ID；  
(2) cms_segid：微信群ID；  
(3) cms_group_id：微信群组ID；  
(4) final_gender_code：性别，值为1是男，为2是女；  
(5) age_level：年龄层级（1、2、3、4）；  
(6) pvalue_level：消费档次；  
(7) shopping_level：购物的层级；  
(8) occupation：是否为大学生；  
(9) new_user_class_level：城市等级；

- behavior_log.csv：涵盖了raw_sample.csv数据中的全部用户22天内的购物行为（共七亿条记录），字段说明如下：  
(1) user_id：脱敏过的用户ID；  
(2) time_stamp：时间戳；  
(3) btag：用户的行为类型，包括四种（浏览|pv、加入购物车|cart、收藏|fav、购买|buy）；  
(4) cate_id：脱敏过的商品类别ID；  
(5) brand_id：脱敏过的品牌ID；  

### 项目的总体流程
本项目是对非搜索类型的广告进行点击率预测和推荐（没有搜索词、没有广告的内容特征信息），总体流程如下：  
- 离线召回模块：利用用户行为日志信息behavior_log.csv，构建用户-类别和用户-品牌的评分数据，训练ALS协同过滤模型，召回Top-N的商品类别，关联对应的广告id缓存到redis中
- 训练排序模型：基于用户的点击日志信息和用户及广告的基本特征信息，离线训练CTR预测模型并保存（LR模型），供线上推荐广告排序使用
- 在线推荐模块：根据输入的用户ID，找到缓存中对应的广告召回候选集和用户及广告的基本特征信息，加载训练好的CTR预测模型进行点击率预测排序，最终返回Top-N推荐结果

### 数据存储
1.先在本地创建一个dataset文件夹，将数据下载到该文件夹下，打开terminal命令行终端，执行如下命令将数据存放到hdfs下：
```
# 先在hdfs创建该项目文件夹recsys，再将数据存放到该目录下
hadoop fs -mkdir /recsys
hadoop fs -put ./dataset /recsys
# 查看是否成功存储
hadoop fs -ls /recsys/dataset
```

### 根据用户行为数据创建ALS模型并召回广告（离线召回模块）
1.创建spark session
```
from pyspark import SparkConf
from pyspark.sql import SparkSession

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
```

2.从hdfs中加载数据为dataframe，并设置结构
```
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

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
```

3.统计每个用户对各个商品类别cate和品牌brand的pv、fav、cart、buy数量，将数据存储到hdfs中，便于后续其他操作使用
```
# 统计每个用户对各个商品类别cate和品牌brand的pv、fav、cart、buy数量
cate_count_df = behavior_log_df.groupBy(behavior_log_df.userId, behavior_log_df.cateId).pivot('btag',
                                        ['pv', 'fav', 'cart', 'buy']).count()
brand_count_df = behavior_log_df.groupBy(behavior_log_df.userId, behavior_log_df.brandId).pivot('btag',
                                         ['pv', 'fav', 'cart', 'buy']).count()
# 将数据存储到hdfs中，便于后续其他操作使用
cate_count_df.write.csv("hdfs://localhost:9000/recsys/preprocessing_dataset/cate_count.csv", header=True)
brand_count_df.write.csv("hdfs://localhost:9000/recsys/preprocessing_dataset/brand_count.csv", header=True)
```

4.根据用户对广告类别cate偏好打分训练ALS模型，并将训练好的模型存储到hdfs中
```
# 设置检查点，防止模型训练造成内存溢出
spark.sparkContext.setCheckpointDir("hdfs://localhost:9000/recsys/checkPoint/")
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
    return r.userId, r.cateId, rating
# 用户对商品类别进行评分
cate_rating_df = cate_count_df.rdd.map(process_row).toDF(["userId", "cateId", "rating"])
# 利用打分数据，训练ALS模型
als = ALS(userCol='userId', itemCol='cateId', ratingCol='rating', checkpointInterval=5)
model = als.fit(cate_rating_df)
# 将训练完成的模型进行存储
model.save("hdfs://localhost:9000/recsys/models/userCateRatingALSModel.obj")
```

5.将模型召回结果缓存到redis中
```
# 设置redis缓存函数
def recall_cate_by_cf(partition):
    # 建立redis连接池和客户端
    pool = redis.ConnectionPool(host='localhost', port=6379, password="123456")
    client = redis.Redis(connection_pool=pool)

    for row in partition:
        client.hset("recall_cate", row.userId, str([i.cateId for i in row.recommendations]))
# 召回用户喜欢的top-5个广告类别
recall_result = als_model.recommendForAllUsers(5)
# 将模型召回结果缓存到redis中
recall_result.foreachPartition(recall_cate_by_cf)
```

6.根据用户喜好的广告类别找到对应的商品进行召回并缓存到redis中
```
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
```

7.同理，根据以上步骤可以进行用户对广告品牌brand偏好打分训练ALS模型，并召回相关的结果进行缓存

### 基于用户的点击日志数据和基本特征信息，离线训练LR点击率预估模型并保存
1.从hdfs中加载raw_sample.csv数据，并对pid字段进行独热编码
```
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
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
```

2.加载用户和广告的基本特征信息，并对pvalue_level和new_user_class_level字段进行独热编码
```
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
```

3.进行数据合并，剔除冗余特征
```
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
```

4.根据特征字段计算特征向量，划分训练集和测试集
```
from pyspark.ml.feature import VectorAssembler
# 根据特征字段计算特征向量
df_data = VectorAssembler().setInputCols(useful_cols[2:]).setOutputCol("features").transform(df_data)
# 根据时间划分数据集，将前7天数据作为训练集，最后一天作为测试集
train_data = df_data.filter(df_data.timestamp <= (1494691186-24*60*60))
test_data = df_data.filter(df_data.timestamp > (1494691186-24*60*60))
```

5.创建LR模型，离线训练并保存
```
from pyspark.ml.classification import LogisticRegression
# 创建逻辑回归模型，进行模型训练并保存模型
lr = LogisticRegression()
model = lr.setLabelCol("clk").setFeaturesCol("features").fit(train_data)
model.save("hdfs://localhost:9000/recsys/models/CTRModel_Normal.obj")
```

### 基于用户召回的候选广告进行LR点击率预测排序，为用户推荐top-n个物品（在线推荐模块）
1.先将用户和广告的基本特征信息缓存到redis中
- 缓存广告特征
```
import json
import redis
def store_ad_redis(partition):
    client = redis.StrictRedis(host="localhost", port=6379, password="123456", db=10)
    for r in partition:
        data = {"price": r.price}
        # 转为json字符串保存
        client.hsetnx("ad_features", r.adgroupId, json.dumps(data))
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
```
- 缓存用户特征
```
def store_user_redis(partition):
    client = redis.StrictRedis(host="localhost", port=6379, password="123456", db=10)
    for r in partition:
        data = {
            "cms_segid": r.cms_segid,
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
```

2.初始化并启动推荐系统，包括创建spark session、建立redis客户端连接、加载训练好的CTR预测模型等操作
```
# 创建spark session
conf = SparkConf()
config = [
    ("spark.app.name", self.SPARK_APP_NAME),
    ("spark.executor.memory", "10g"),
    ("spark.master", self.SPARK_URL),
    ("spark.executor.cores", "3")
]
conf.setAll(config)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# 建立redis客户端连接
client_of_recall = redis.StrictRedis(host="localhost", port=6379, password="123456", db=9)
client_of_features = redis.StrictRedis(host="localhost", port=6379, password="123456", db=10)

from pyspark.ml.classification import LogisticRegressionModel
# 加载训练好的CTR预测模型
CTR_model = LogisticRegressionModel.load(self.MODEL_PATH)
```

3.根据输入的用户ID和广告资源位置ID，创建模型的输入数据集
```
def create_dataset(self, user_id, pid):
    """创建LR排序模型的输入数据集"""
    # 从redis缓存中获取用户特征和召回物品
    user_feature = json.loads(self.client_of_features.hget("user_features", user_id).decode())
    recall_sets = self.client_of_recall.smembers(user_id)
    result = []

    # 遍历召回物品集合
    for adgroupId in recall_sets:
        # 获取广告特征
        adgroupId = int(adgroupId)
        ad_feature = json.loads(self.client_of_features.hget("ad_features", adgroupId).decode())
        # 合并用户和广告特征
        features = {}
        features.update(user_feature)
        features.update(ad_feature)
        for k, v in features.items():
            if v is None:
                features[k] = -1

        features_col = ["price", "cms_segid", "cms_group_id", "final_gender_code", "age_level", "shopping_level",
                        "occupation", "pid", "pvalue_level", "new_user_class_level"]
        price = float(features["price"])
        cms = features["cms_segid"]
        cms_group_id = features["cms_group_id"]
        final_gender_code = features["final_gender_code"]
        age_level = features["age_level"]
        shopping_level = features["shopping_level"]
        occupation = features["occupation"]

        pid_value = [0 for i in range(2)]
        pvalue_level_value = [0 for i in range(4)]
        new_user_class_level_value = [0 for i in range(5)]
        pid_value[self.pid_rela[pid]] = 1
        pvalue_level_value[self.pvalue_level_rela[int(features["pvalue_level"])]] = 1
        new_user_class_level_value[self.new_user_class_level_rela[int(features["new_user_class_level"])]] = 1

        vector = DenseVector(pid_value + [price, cms, cms_group_id, final_gender_code, age_level, shopping_level,
                             occupation] + pvalue_level_value + new_user_class_level_value)

        result.append((user_id, adgroupId, vector))
    return result
```

4.利用训练好的模型预测广告的点击率，返回用户最有可能点击的前N个广告ID
```
pdf = pd.DataFrame(create_dataset(user_id, pid), columns=["userId", "adgroupId", "features"])
dataset = spark.createDataFrame(pdf)
prediction = CTR_model.transform(dataset).sort("probability")
result = [i.adgroupId for i in prediction.select("adgroupId").head(n)]  # n为需要推荐的广告个数
```

5.项目最终命令行运行效果如下：
```
推荐系统正在启动...
推荐系统启动成功.
请输入用户ID：8
请输入广告资源位（输入1（默认）：430548_1007 | 输入2：430549_1007）：1
请输入需要推荐的广告个数（默认10，最大500）：10
给用户8推荐的广告ID列表为： [525832, 417674, 832203, 510906, 372559, 194931, 484955, 298084, 548724, 597820]
继续请输入1，否则输入其他任意键退出：1
请输入用户ID：268
请输入广告资源位（输入1（默认）：430548_1007 | 输入2：430549_1007）：2
请输入需要推荐的广告个数（默认10，最大500）：5
给用户268推荐的广告ID列表为： [510344, 192802, 42109, 754158, 305992]
继续请输入1，否则输入其他任意键退出：
```
