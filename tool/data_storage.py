"""
@Author: YMH
@Date: 2022-4-22
@Description: 将从阿里云天池大赛下载的商品数据存储到hbase和mongodb等数据库中，便于后续项目使用
"""

import happybase
import pymongo
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
import json

HDFS_PATH = "hdfs://localhost:9000/recsys/dataset/"
SPARK_APP_NAME = "ALSRecommend"
SPARK_URL = "spark://ymh:7077"

# 建立mongodb数据库连接
mongo_client = pymongo.MongoClient(host='localhost', port=27017)
db = mongo_client['recsys']


def store_data_hbase():
    connection = happybase.Connection(host='localhost', port=9090)


if __name__ == "__main__":
    # connection = happybase.Connection(host='localhost', port=9090)
    # # connection.create_table('raw_sample', {'click_info': dict(max_versions=3)})
    # # connection.create_table('ad_feature', {'base_info': dict(max_versions=3)})
    # # connection.create_table('user_profile', {'base_info': dict(max_versions=3)})
    # # connection.create_table('raw_behavior_log', {'behavior_info': dict(max_versions=3)})
    # table_list = connection.tables()
    # print(table_list)
    # 建立spark sql连接
    config = (
        ("spark.app.name", SPARK_APP_NAME),
        ("spark.executor.memory", "6g"),
        ("spark.master", SPARK_URL),
        ("spark.executor.cores", "3"),
    )
    conf = SparkConf()
    conf.setAll(config)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # file_list = ['ad_feature.csv', 'user_profile.csv', 'raw_sample.csv', 'behavior_log.csv']
    file_list = ['raw_sample.csv', 'behavior_log.csv']
    for file in file_list:
        collection = db[file[:-4]]
        spark_df = spark.read.csv(HDFS_PATH + file, header=True)
        df = spark_df.toPandas()
        collection.insert_many(json.loads(df.T.to_json()).values())
        print(file)

