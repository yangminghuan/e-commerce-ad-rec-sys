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

### 根据用户行为数据创建ALS模型并召回广告
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

6.同理，根据以上步骤可以进行用户对广告品牌brand偏好打分训练ALS模型，并召回相关的结果进行缓存


